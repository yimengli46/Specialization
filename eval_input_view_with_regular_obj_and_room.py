import torch.optim as optim
import os
import numpy as np
from modeling.utils.ResNet_multilabel import mlp, mlp_wo_room_type
from sseg_utils.saver import Saver
from sseg_utils.summaries import TensorboardSummary
import matplotlib.pyplot as plt
from dataloader_regular_obj_and_room import get_all_view_dataset, my_collate
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


def gen_plot(y_pred, y_label):
    """Create a pyplot plot and save to buffer."""
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 2))

    y_pred = (np.array(y_pred).flatten() > 0.5)
    y_label = np.array(y_label).flatten()
    acc = (y_pred == y_label).mean()
    f1 = f1_score(y_label, y_pred, average='weighted')

    y_pred = np.reshape(y_pred, (-1, 6))
    y_label = np.reshape(y_label, (-1, 6))

    idx0 = random.choice((range(y_pred.shape[0])))
    y_pred = np.reshape(y_pred[idx0], (1, -1))
    y_label = np.reshape(y_label[idx0], (1, -1))

    ax[0].imshow(y_pred)
    ax[0].set_title(f'pred, acc = {acc:.2}, f1 = {f1:.2}')
    ax[1].imshow(y_label)
    ax[1].set_title('label')
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    fig.tight_layout()
    return fig  # buf


# def eval(model_type):
model_type = 'mlp_wo_room_type'  # 'knowledge_graph'  # 'context_matrix'  # 'resnet'
if model_type == 'mlp':
    trained_model_dir = '/home/yimeng/ARGO_scratch/topo_map_specialization/Specialization/output/model_weights_input_view/mlp/experiment_9/best_checkpoint.pth.tar'
elif model_type == 'mlp_wo_room_type':
    trained_model_dir = '/home/yimeng/ARGO_scratch/topo_map_specialization/Specialization/output/model_weights_input_view/mlp_wo_room_type/experiment_0/best_checkpoint.pth.tar'


if model_type == 'mlp':
    cfg.merge_from_file(
        'configs/exp_train_input_view_model_MLP.yaml')
elif model_type == 'mlp_wo_room_type':
    cfg.merge_from_file(
        'configs/exp_train_input_view_model_MLP_wo_room_type.yaml')
else:
    raise NotImplementedError("This model is not implemented.")
cfg.freeze()

output_folder = cfg.PRED.VIEW.SAVED_FOLDER
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

saver = Saver(output_folder)

cfg.dump(stream=open(
    f'{saver.experiment_dir}/experiment_config.yaml', 'w'))

# ============================================ Define Tensorboard Summary =================================
summary = TensorboardSummary(saver.experiment_dir)

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
    'val', data_folder, hm3d_to_lvis_dict, LVIS_dict, test_transform)
dataloader_val = data.DataLoader(dataset_val,
                                 batch_size=cfg.PRED.VIEW.BATCH_SIZE,
                                 num_workers=cfg.PRED.VIEW.NUM_WORKERS,
                                 shuffle=False,
                                 collate_fn=my_collate,
                                 )

# ================================================================================================================================

# Define network
if cfg.PRED.VIEW.MODEL_TYPE == 'mlp':
    model = mlp()
elif cfg.PRED.VIEW.MODEL_TYPE == 'mlp_wo_room_type':
    model = mlp_wo_room_type()
model = nn.DataParallel(model)
model = model.cuda()

# =========================================================== Define Optimizer ================================================
train_params = [{'params': model.parameters(), 'lr': cfg.PRED.VIEW.LR}]
optimizer = optim.Adam(
    train_params, lr=cfg.PRED.VIEW.LR, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Define Criterion
# whether to use class balanced weights
weight = None

criterion = nn.BCEWithLogitsLoss()
best_test_loss = 1e10

# ===================================================== Resuming checkpoint ====================================================
best_pred = 0.0
if trained_model_dir != '':
    if not os.path.isfile(trained_model_dir):
        raise RuntimeError(
            "=> no checkpoint found at '{}'" .format(trained_model_dir))
    checkpoint = torch.load(trained_model_dir)
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}')".format(trained_model_dir))

# ======================================================== evaluation stage =====================================================

# if epoch % cfg.PRED.VIEW.EVAL_INTERVAL == 0:
model.eval()
test_loss = []
iter_num = 0

with torch.no_grad():
    y_pred = np.zeros((0, 6))
    y_label = np.zeros((0, 6))
    y_de_or_vi = np.zeros((0, 6))

    count_vis_img = 0

    for idx_dl, batch in enumerate(dataloader_val):
        print('iter_num = {}'.format(iter_num))
        inputs, gt_outputs, batch_de_or_vi = batch['input'], batch['output'], batch['detected_or_vicinity']
        print(f'inputs.shape = {inputs.shape}')
        print(f'gt_outputs.shape = {gt_outputs.shape}')

        A, B, _ = inputs.shape
        inputs = inputs.view(A * B, -1)
        gt_outputs = gt_outputs.view(A * B)

        inputs = inputs.cuda()
        gt_outputs = gt_outputs.cuda()

        # ================================================ compute loss =============================================

        if cfg.PRED.VIEW.MODEL_TYPE == 'mlp':
            # batchsize x 1 x H x W
            output = model(inputs)
        elif cfg.PRED.VIEW.MODEL_TYPE == 'mlp_wo_room_type':
            output = model(inputs)
        print(f'output.shape = {output.shape}')

        loss = criterion(output, gt_outputs.float())

        # concatenate the results
        output = output.view(A, B).cpu().numpy()
        gt_outputs = gt_outputs.view(A, B).cpu().numpy()
        batch_de_or_vi = batch_de_or_vi.numpy()

        y_pred = np.concatenate((y_pred, output), axis=0)
        y_label = np.concatenate((y_label, gt_outputs), axis=0)
        y_de_or_vi = np.concatenate((y_de_or_vi, batch_de_or_vi), axis=0)

        '''
        # visualize the results
        batch_rgb = batch['original_img']
        for i in range(A):
            rgb = batch_rgb[i]
            de_or_vi = batch_de_or_vi[i]

            item_output = (output[i] > 0)
            item_gt = (gt_outputs[i] > 0)

            if not np.all(item_output == item_gt) and (2 in de_or_vi):
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
                ax.imshow(rgb)
                ax.set_title(f'label: {item_gt}\nmyres: {item_output}\n {de_or_vi}')
                sample_name = str(idx_dl).zfill(4)
                fig.savefig(f'output/vis_obj_room_infer_results/{sample_name}_{i}.jpg',
                            bbox_inches='tight')
                plt.close()
        '''

        iter_num += 1

# Fast test during the training
aps = []
aps_vi = []
for idx_class in range(0, y_label.shape[1]):
    class_label = y_label[:, idx_class]
    class_pred = y_pred[:, idx_class]
    class_de_or_vi = y_de_or_vi[:, idx_class]
    if class_label.sum() > 0:
        mask = (class_de_or_vi == 0) | (class_de_or_vi == 3)
        ap = average_precision_score(
            class_label[mask], class_pred[mask])
        print(
            f'-------  Class: {idx_class}     AP: {ap:.4f}  -------')
        aps.append(ap)

        mask = (class_de_or_vi == 0) | (class_de_or_vi == 2)
        ap_vi = average_precision_score(
            class_label[mask], class_pred[mask])
        print(
            f'-------vi  Class: {idx_class}     AP: {ap_vi:.4f}  -------')
        aps_vi.append(ap_vi)


mAP = np.mean(aps)
mAP_vi = np.mean(aps_vi)

print('Validation:')
print(f'numImages: {len(dataloader_val)}')
print(f'mAP: {mAP:.3f}')
print(f'mAP_vi: {mAP_vi:.3f}')
