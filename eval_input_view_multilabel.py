import torch.optim as optim
import os
import numpy as np
from modeling.utils.ResNet_multilabel import resnet, context_matrix, cnn, knowledge_graph, cm_and_kg
from sseg_utils.saver import Saver
from sseg_utils.summaries import TensorboardSummary
import matplotlib.pyplot as plt
from dataloader_input_view_by_densely_sampled_locations import get_all_view_dataset, my_collate
import torch.utils.data as data
import torch
import torch.nn as nn
from core import cfg
import bz2
import _pickle as cPickle
import argparse
from sklearn.metrics import f1_score, average_precision_score
import random
import torch.nn.functional as F
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(1024),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])


# def test(model_type):
model_type = 'knowledge_graph'  # 'knowledge_graph'  # 'context_matrix'  # 'resnet'

if model_type == 'context_matrix':
    checkpoint_file = 'output/model_weights_input_view/context_matrix/experiment_3/best_checkpoint.pth.tar'
elif model_type == 'knowledge_graph':
    checkpoint_file = 'output/model_weights_input_view/knowledge_graph/experiment_13/best_checkpoint.pth.tar'

bbox_type = 'Detic'  # Detic, gt

if model_type == 'resnet':
    cfg.merge_from_file(
        'configs/exp_train_input_view_multilabel_model_resnet.yaml')
elif model_type == 'knowledge_graph':
    cfg.merge_from_file(
        'configs/exp_train_input_view_multilabel_model_knowledge_graph.yaml')
elif model_type == 'context_matrix':
    cfg.merge_from_file(
        'configs/exp_train_input_view_multilabel_model_context_matrix.yaml')
elif model_type == 'clip':
    cfg.merge_from_file(
        'configs/exp_train_input_view_model_clip.yaml')
elif model_type == 'cnn':
    cfg.merge_from_file(
        'configs/exp_train_input_view_multilabel_model_cnn.yaml')
elif model_type == 'cm_and_kg':
    cfg.merge_from_file(
        'configs/exp_train_input_view_multilabel_model_combine_context_matrix_and_kg.yaml')
else:
    raise NotImplementedError("This model is not implemented.")
cfg.freeze()

output_folder = cfg.PRED.VIEW.SAVED_FOLDER
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# ============================================== load necessary data ============================
hm3d_to_lvis_dict = np.load(
    'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

# load the LVIS categories
with bz2.BZ2File(f'output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
    LVIS_dict = cPickle.load(fp)
    lvis_cat_synonyms_list = LVIS_dict['cat_synonyms']
    lvis_rowid_to_catid_dict = LVIS_dict['rowid2catid_dict']
    lvis_cat_synonyms_embedding = LVIS_dict['cat_synonyms_embedding']
    lvis_cat_index_embedding = LVIS_dict['cat_id_embedding']

# get index of all the goal objs
goal_obj_list = sorted(list(set(hm3d_to_lvis_dict.values())))  # size: 351
goal_obj_index_list = list(set(lvis_rowid_to_catid_dict[lvis_cat_synonyms_list.index(
    cat_syn)] for cat_syn in goal_obj_list))  # size: 310
goal_obj_index_embeddings = lvis_cat_index_embedding[[
    i - 1 for i in goal_obj_index_list]]  # shape: 310 x 384

# =========================================================== Define Dataloader ==================================================
data_folder = cfg.PRED.VIEW.DENSELY_SAMPLED_LOCATIONS_SAVED_FOLDER

dataset_val = get_all_view_dataset(
    'val', data_folder, hm3d_to_lvis_dict, LVIS_dict, test_transform, bbox_type)
dataloader_val = data.DataLoader(dataset_val,
                                 batch_size=cfg.PRED.VIEW.BATCH_SIZE,
                                 num_workers=cfg.PRED.VIEW.NUM_WORKERS,
                                 shuffle=False,
                                 collate_fn=my_collate,
                                 )

# ================================================================================================================================

# Define network
if cfg.PRED.VIEW.MODEL_TYPE == 'resnet':
    model = resnet(cfg.PRED.VIEW.RESNET_INPUT_CHANNEL,
                   cfg.PRED.VIEW.RESNET_OUTPUT_CHANNEL)
elif cfg.PRED.VIEW.MODEL_TYPE == 'context_matrix':
    model = context_matrix(lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                           goal_obj_index_list, goal_obj_index_embeddings)
elif cfg.PRED.VIEW.MODEL_TYPE == 'knowledge_graph':
    model = knowledge_graph(lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                            goal_obj_index_list, goal_obj_index_embeddings)
elif cfg.PRED.VIEW.MODEL_TYPE == 'cnn':
    model = cnn(cfg.PRED.VIEW.INPUT_CHANNEL,
                cfg.PRED.VIEW.OUTPUT_CHANNEL)
elif cfg.PRED.VIEW.MODEL_TYPE == 'cm_and_kg':
    model = cm_and_kg(lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                      goal_obj_index_list, goal_obj_index_embeddings)
model = nn.DataParallel(model)
model = model.cuda()

# ===================================================== Resuming checkpoint ====================================================
if checkpoint_file != '':
    if not os.path.isfile(checkpoint_file):
        raise RuntimeError(
            "=> no checkpoint found at '{}'" .format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> loaded checkpoint {checkpoint_file}")

# ======================================================== evaluation stage =====================================================
model.eval()
iter_num = 0

with torch.no_grad():
    y_pred = np.zeros((0, 310))
    y_label = np.zeros((0, 310))
    original_dist_all = np.zeros((0, 310))
    detector_pred_all = np.zeros((0, 310))

    sampled_batch_ids = random.choices(
        range(len(dataloader_val)), k=10)
    count_vis_img = 0

    for idx_dl, batch in enumerate(dataloader_val):
        print('iter_num = {}'.format(iter_num))
        images, bbox_list, goal_objs, dists, original_dist, detector_pred = batch['rgb'], batch[
            'bbox'], batch['goal_obj'], batch['dist'], batch['original_dist'], batch['detector_pred']
        print(f'images.shape = {images.shape}')
        print(f'dists.shape = {dists.shape}')

        images = images.cuda()
        dists = dists.cuda()

        # ================================================ compute loss =============================================

        if cfg.PRED.VIEW.MODEL_TYPE == 'resnet':
            # batchsize x 1 x H x W
            output = model(images, goal_objs)
        elif cfg.PRED.VIEW.MODEL_TYPE == 'context_matrix':
            output = model(bbox_list, goal_objs)
        elif cfg.PRED.VIEW.MODEL_TYPE == 'knowledge_graph':
            output = model(bbox_list, goal_objs)
        elif cfg.PRED.VIEW.MODEL_TYPE == 'clip':
            # batchsize x 1 x H x W
            output = model(images, goal_objs)
        elif cfg.PRED.VIEW.MODEL_TYPE == 'cnn':
            output = model(images)  # batchsize x 1 x H x W
        elif cfg.PRED.VIEW.MODEL_TYPE == 'cm_and_kg':
            output = model(bbox_list, goal_objs)
        print(f'output.shape = {output.shape}')

        # concatenate the results
        output = output.cpu().numpy()
        dists = dists.cpu().numpy()

        y_pred = np.concatenate((y_pred, output), axis=0)
        y_label = np.concatenate((y_label, dists), axis=0)
        original_dist_all = np.concatenate(
            (original_dist_all, original_dist), axis=0)
        detector_pred_all = np.concatenate(
            (detector_pred_all, detector_pred), axis=0)

        iter_num += 1

aps = []
for idx_class in range(0, y_label.shape[1]):
    class_label = y_label[:, idx_class]
    class_pred = y_pred[:, idx_class]
    if class_label.sum() > 0:
        ap = average_precision_score(
            class_label, class_pred)
        print(
            f'-------  Class: {idx_class}     AP: {ap:.4f}  -------')
        aps.append(ap)

mAP = np.mean(aps)
print(f'mAP = {mAP}')

# save the results
results_dict = {}
results_dict['y_pred'] = y_pred
results_dict['y_label'] = y_label
results_dict['original_dist'] = original_dist_all
results_dict['detector_pred'] = detector_pred_all
with bz2.BZ2File(f'output/multilabel_pred_results_model_{model_type}_bbox_{bbox_type}.pbz2', 'w') as fp:
    cPickle.dump(
        results_dict,
        fp
    )

'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='resnet')
    args = parser.parse_args()

    test(args.model)


if __name__ == "__main__":
    main()
'''
