"""
Read in a frontier image.
Build the embedding from different variations.
"""
import numpy as np
import torch.nn as nn
import torch

import bz2
import _pickle as cPickle
import matplotlib.pyplot as plt
from modeling.utils.baseline_utils import apply_color_to_map
from modeling.utils.ResNet import resnet, context_matrix, knowledge_graph, clip_fc
import skimage.measure
import matplotlib.patches as patches
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image


def get_img_coordinates(img):
    H, W = img.shape[:2]
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack((xv, yv), axis=2)
    return coords


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


split = 'train'
scene_name = '00009-vLpv2VX547B'
floor_id = 0
step_id = 10
# 'knowledge_graph'  # 'context_matrix'  # 'resnet'
embedding_type = 'clip'
target_cat = 'toy'

data_folder = 'output/training_data_input_view_1000samples/train'

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

# load the category id to name dict
idx2cat_dict = np.load(
    f'output/semantic_map/{split}/{scene_name}_{floor_id}/category_id_to_name_dict.npy', allow_pickle=True).item()
# convert hm3d category into goal categories
# goal category set is specific to each scene
# as we only sample goal objects exist in this scene
goal_category_set = set()
ignored_category_id_set = set()
for k, v in idx2cat_dict.items():
    v = v.strip().lower()
    if v in list(hm3d_to_lvis_dict.keys()):
        lvis_goal_obj = hm3d_to_lvis_dict[v]
        idx2cat_dict[k] = lvis_goal_obj
        goal_category_set.add(lvis_goal_obj)
    #     print(f'v = {v}, new_v = {idx2cat_dict[k]}')
    else:
        idx2cat_dict[k] = v
        ignored_category_id_set.add(k)
    #     print(f'class {v} is not a goal object.')

# ================= read the explored map ====================
with bz2.BZ2File(f'{data_folder}/{scene_name}_{floor_id}/{step_id}.pbz2', 'rb') as fp:
    pk_file = cPickle.load(fp)
    cur_sem_map = pk_file['sseg_map']
    cur_occ_map = pk_file['occ_map']
    frontiers = pk_file['frontiers']

# for idx in range(len(frontiers)):
idx = 10
print(f'idx = {idx}')
fron = frontiers[idx]
rgb_img = fron['rgb']
depth_img = fron['depth']
sseg_img = fron['sseg']

'''
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
ax[0].imshow(rgb_img)
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[0].set_title("rgb")
ax[1].imshow(apply_color_to_map(
    sseg_img, type_categories='LVIS'))
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[1].set_title("sseg")
ax[2].imshow(depth_img)
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title("depth")
fig.tight_layout()
plt.show()
'''

# put the image through resnet
if embedding_type == 'resnet':
    tensor_img = torch.tensor(
        rgb_img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    print(f'tensor_img.shape = {tensor_img.shape}')
    model = resnet(3, 1, lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                   goal_obj_index_list, goal_obj_index_embeddings)
    model = model.cuda()
    with torch.no_grad():
        embedding = model(tensor_img, [target_cat])
elif embedding_type == 'clip':
    # tensor_img = torch.tensor(
    #     rgb_img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    preprocess = _transform(224)
    rgb_img = Image.fromarray(rgb_img)
    tensor_img = preprocess(rgb_img).unsqueeze(0).cuda()
    #assert 1 == 2
    model = clip_fc()
    model = model.cuda()
    with torch.no_grad():
        embedding = model(tensor_img, [target_cat])

elif embedding_type == 'context_matrix':
    model = context_matrix(lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                           goal_obj_index_list, goal_obj_index_embeddings)
    model = model.cuda()

    img_hm3d_cat_index_list = np.unique(sseg_img)
    bbox_list = []
    # create image coordinates
    coords = get_img_coordinates(sseg_img)
    # convert sseg_img into bbox
    for hm3d_cat_index in img_hm3d_cat_index_list:
        lvis_cat_name = idx2cat_dict[hm3d_cat_index]
        # if current category is in the goal category list
        if lvis_cat_name in goal_category_set:
            print(f'cat_name = {lvis_cat_name}')
            # create binary image for current category index
            cat_binary_map = np.zeros(sseg_img.shape, dtype=np.int16)
            cat_binary_map[sseg_img == hm3d_cat_index] = 1
            instance_label, num_ins = skimage.measure.label(
                cat_binary_map, background=0, connectivity=1, return_num=True)
            #print(f'num_ins = {num_ins}')
            # plt.imshow(cat_binary_map)
            # plt.show()
            # create a bbox for each mask
            for idx_ins in range(1, num_ins + 1):
                mask_ins = (instance_label == idx_ins)
                mask_coords = coords[mask_ins]
                x1 = np.min(mask_coords[:, 0])
                x2 = np.max(mask_coords[:, 0])
                y1 = np.min(mask_coords[:, 1])
                y2 = np.max(mask_coords[:, 1])
                cat_id = lvis_rowid_to_catid_dict[lvis_cat_synonyms_list.index(
                    lvis_cat_name)]
                if x2 - x1 > 5 and y2 - y1 > 5:
                    bbox_list.append([x1, y1, x2, y2, cat_id])

    # plot the bboxes
    fig, ax = plt.subplots()
    ax.imshow(rgb_img)
    for bbox in bbox_list:
        x1, y1, x2, y2, _ = bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

    # compute context matrix
    with torch.no_grad():
        embedding = model([bbox_list], [target_cat])

elif embedding_type == 'knowledge_graph':
    model = knowledge_graph(lvis_cat_synonyms_list, lvis_cat_synonyms_embedding,
                            goal_obj_index_list, goal_obj_index_embeddings)
    model = model.cuda()

    img_hm3d_cat_index_list = np.unique(sseg_img)
    bbox_list = []
    # create image coordinates
    coords = get_img_coordinates(sseg_img)
    # convert sseg_img into bbox
    for hm3d_cat_index in img_hm3d_cat_index_list:
        lvis_cat_name = idx2cat_dict[hm3d_cat_index]
        # if current category is in the goal category list
        if lvis_cat_name in goal_category_set:
            print(f'cat_name = {lvis_cat_name}')
            # create binary image for current category index
            cat_binary_map = np.zeros(sseg_img.shape, dtype=np.int16)
            cat_binary_map[sseg_img == hm3d_cat_index] = 1
            instance_label, num_ins = skimage.measure.label(
                cat_binary_map, background=0, connectivity=1, return_num=True)
            #print(f'num_ins = {num_ins}')
            # plt.imshow(cat_binary_map)
            # plt.show()
            # create a bbox for each mask
            for idx_ins in range(1, num_ins + 1):
                mask_ins = (instance_label == idx_ins)
                mask_coords = coords[mask_ins]
                x1 = np.min(mask_coords[:, 0])
                x2 = np.max(mask_coords[:, 0])
                y1 = np.min(mask_coords[:, 1])
                y2 = np.max(mask_coords[:, 1])
                cat_id = lvis_rowid_to_catid_dict[lvis_cat_synonyms_list.index(
                    lvis_cat_name)]
                if x2 - x1 > 5 and y2 - y1 > 5:
                    bbox_list.append([x1, y1, x2, y2, cat_id])

    # plot the bboxes
    fig, ax = plt.subplots()
    ax.imshow(rgb_img)
    for bbox in bbox_list:
        x1, y1, x2, y2, _ = bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

    # compute context matrix
    with torch.no_grad():
        embedding = model([bbox_list], [target_cat])
