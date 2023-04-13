import torch.optim as optim
import os
import numpy as np
from modeling.utils.ResNet import resnet, context_matrix, knowledge_graph, clip_fc
from sseg_utils.saver import Saver
from sseg_utils.summaries import TensorboardSummary
import matplotlib.pyplot as plt
from dataloader_input_view import get_all_view_dataset, my_collate
import torch.utils.data as data
import torch
import torch.nn as nn
from core import cfg
import bz2
import _pickle as cPickle
import argparse
from sklearn.metrics import f1_score
import random
import torch.nn.functional as F


def gen_plot(y_pred, y_label):
    """Create a pyplot plot and save to buffer."""
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 2))

    y_pred = (np.array(y_pred).flatten() > 0)
    y_label = np.array(y_label).flatten()
    acc = (y_pred == y_label).mean()
    f1 = f1_score(y_label, y_pred, average='weighted')

    y_pred = np.reshape(y_pred, (-1, 310))
    y_label = np.reshape(y_label, (-1, 310))

    idx0 = random.choice((range(y_pred.shape[0])))
    y_pred = np.reshape(y_pred[idx0], (1, -1))
    y_label = np.reshape(y_label[idx0], (1, -1))

    ax[0].imshow(y_pred)
    ax[0].set_title(f'pred, acc = {acc:.2}, f1 = {f1:.2}')
    ax[1].imshow(y_label)
    ax[1].set_title('label')
    #buf = io.BytesIO()
    #plt.savefig(buf, format='png')
    # buf.seek(0)
    fig.tight_layout()
    return fig  # buf


def focal_loss(x, y):
    '''Focal loss.

    Args:
        x: (tensor) sized [N,D].
        y: (tensor) sized [N,].

    Return:
        (tensor) focal loss.
    '''
    alpha = 0.25
    gamma = 2

    t = y  # F.one_hot(y, num_classes=2)
    #print(f't = {t}')

    p = x.sigmoid()
    pt = p * t + (1 - p) * (1 - t)         # pt = p if t > 0 else 1-p
    w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1 - pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t.float(), w.detach(), reduction='mean')


model_type = 'knowledge_graph'
experiment_id = '7'
model_saved_folder = f'output/model_weights_input_view/{model_type}/experiment_{experiment_id}/checkpoint.pth.tar'

if model_type == 'resnet':
    cfg.merge_from_file('configs/exp_train_input_view_model_resnet.yaml')
elif model_type == 'knowledge_graph':
    cfg.merge_from_file(
        'configs/exp_train_input_view_model_knowledge_graph.yaml')
elif model_type == 'context_matrix':
    cfg.merge_from_file(
        'configs/exp_train_input_view_model_context_matrix.yaml')
elif model_type == 'clip':
    cfg.merge_from_file(
        'configs/exp_train_input_view_model_clip.yaml')
else:
    raise NotImplementedError("This model is not implemented.")
cfg.freeze()


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
data_folder = cfg.PRED.VIEW.PROCESSED_VIEW_SAVED_FOLDER
dataset_val = get_all_view_dataset(
    'val', data_folder, hm3d_to_lvis_dict, LVIS_dict)
dataloader_val = data.DataLoader(dataset_val,
                                 batch_size=64,
                                 num_workers=20,
                                 shuffle=False,
                                 collate_fn=my_collate,
                                 )

# ================================================================================================================================

# Define network
if cfg.PRED.VIEW.MODEL_TYPE == 'resnet':
    model = resnet(cfg.PRED.VIEW.RESNET_INPUT_CHANNEL,
                   cfg.PRED.VIEW.RESNET_OUTPUT_CHANNEL,
                   lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                   goal_obj_index_list, goal_obj_index_embeddings)
elif cfg.PRED.VIEW.MODEL_TYPE == 'context_matrix':
    model = context_matrix(lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                           goal_obj_index_list, goal_obj_index_embeddings)
elif cfg.PRED.VIEW.MODEL_TYPE == 'knowledge_graph':
    model = knowledge_graph(lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                            goal_obj_index_list, goal_obj_index_embeddings)
elif cfg.PRED.VIEW.MODEL_TYPE == 'clip':
    model = clip_fc()
model = nn.DataParallel(model)
model = model.cuda()

# ===================================================== Resuming checkpoint ====================================================
best_pred = 0.0
if model_saved_folder != '':
    if not os.path.isfile(model_saved_folder):
        raise RuntimeError(
            "=> no checkpoint found at '{}'" .format(model_saved_folder))
    checkpoint = torch.load(model_saved_folder)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(
        model_saved_folder, checkpoint['epoch']))


# ======================================================== evaluation stage =====================================================

model.eval()
test_loss = 0.0
iter_num = 0

y_pred = []
y_label = []

sampled_batch_ids = random.choices(
    range(len(dataloader_val)), k=10)
count_vis_img = 0

for idx_dl, batch in enumerate(dataloader_val):
    print('iter_num = {}'.format(iter_num))
    images, bbox_list, goal_objs, dists = batch['rgb'], batch[
        'bbox'], batch['goal_obj'], batch['dist']
    print(f'images.shape = {images.shape}')
    print(f'dists.shape = {dists.shape}')

    images = images.cuda()
    dists = dists.cuda()

    # ================================================ compute loss =============================================
    with torch.no_grad():
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

    print(f'output.shape = {output.shape}')
    B, C = output.shape[:2]
    output = output.view(-1)
    dists = dists.view(-1)
    loss = focal_loss(output, dists)

    # concatenate the results
    output = (output > 0).cpu().tolist()
    dists = dists.cpu().tolist()

    y_pred += output
    y_label += dists

    if idx_dl in sampled_batch_ids:
        output = np.array(output)
        dists = np.array(dists)
        fig = gen_plot(output, dists)
        count_vis_img += 1

    test_loss += loss.item()
    print('Test loss: %.3f' % (test_loss / (iter_num + 1)))

    iter_num += 1


# compuate acc
y_pred = (np.array(y_pred).flatten() > 0)
y_label = np.array(y_label).flatten()
acc = (y_pred == y_label).mean()
f1 = f1_score(y_label, y_pred, average='weighted')

pred_dict = {}
pred_dict['y_pred'] = y_pred
pred_dict['y_label'] = y_label
np.save(f'{model_type}_pred_dict.npy', pred_dict)
