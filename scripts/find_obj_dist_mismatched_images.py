'''
find the mis-alignment between the gt detections and dist computed from maps
'''

import numpy as np
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
import pandas as pd
from dataloader_input_view_by_densely_sampled_locations import process_lvis_dict
from dataloader_input_view_by_densely_sampled_locations import get_all_view_dataset, my_collate

from PIL import Image, ImageDraw, ImageFont
import skimage.measure
import matplotlib.patches as patches


def get_img_coordinates(img):
    H, W = img.shape[:2]
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack((xv, yv), axis=2)
    return coords


bbox_type = 'gt'  # 'Detic', 'gt'
model_type = 'context_matrix'
saved_folder = 'detector_analysis'
thresh_detector = 0.3

# ================================= load the datasets ============================
hm3d_to_lvis_dict = np.load(
    'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

# load the LVIS categories
with bz2.BZ2File('output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
    LVIS_dict = cPickle.load(fp)
    lvis_cat_synonyms_list = LVIS_dict['cat_synonyms']
    lvis_rowid_to_catid_dict = LVIS_dict['rowid2catid_dict']
    lvis_cat_synonyms_embedding = LVIS_dict['cat_synonyms_embedding']
    lvis_cat_index_embedding = LVIS_dict['cat_id_embedding']

goal_obj_list, goal_obj_index_list, lvis_cat_name_to_lvis_id_dict, lvis_id_to_lvis_cat_names_dict = process_lvis_dict(
    hm3d_to_lvis_dict, LVIS_dict)

# =============================== read the results =================================
with bz2.BZ2File(f'output/{saved_folder}/multilabel_pred_results_model_{model_type}_bbox_{bbox_type}.pbz2', 'rb') as fp:
    results_dict = cPickle.load(fp)

y_pred = results_dict['y_pred']
y_label = results_dict['y_label']
original_dist_all = results_dict['original_dist']
detector_pred_all = results_dict['detector_pred']

# =========================================================== Define Dataloader ==================================================
data_folder = 'output/training_data_input_view_by_densely_sample_locations_FULL_SIZE'

dataset_val = get_all_view_dataset(
    'val', data_folder, hm3d_to_lvis_dict, LVIS_dict, None, bbox_type)

idx2cat_dict = np.load(
    f'output/semantic_map/val/00800-TEEsavR23oF_0/category_id_to_name_dict.npy', allow_pickle=True).item()

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

# ============================ analyze the data =========================
num_images, num_classes = y_pred.shape
count = 0
for idx_img in range(num_images):
    original_dist = original_dist_all[idx_img, :]

    label_in_the_view = (original_dist == 1)
    label_near_the_view = (original_dist == 2)

    detector_class = detector_pred_all[idx_img, :] > thresh_detector

    if not np.array_equal(label_in_the_view, detector_class):
        set_map_true_indices = set(np.where(label_in_the_view)[0])
        set_detector_true_indices = set(np.where(detector_class)[0])
        set_map_near_indices = set(np.where(label_near_the_view)[0])

        # objs showed in detector but not showed on map
        indices_detector = set_detector_true_indices - set_map_true_indices
        str_detector = 'detector: '
        for idx in indices_detector:
            str_detector += lvis_id_to_lvis_cat_names_dict[goal_obj_index_list[idx]][0]
            str_detector += ', '

        # objs showed on map but not showed in detector
        indices_map = set_map_true_indices - set_detector_true_indices
        str_map = 'map: '
        for idx in indices_map:
            str_map += lvis_id_to_lvis_cat_names_dict[goal_obj_index_list[idx]][0]
            str_map += ', '

        print(f'count = {count}')
        img_addr = dataset_val[idx_img]
        image = Image.open(f'{img_addr}.jpg')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
        ax.imshow(image)
        ax.text(10, 10, str_detector, color='blue')
        ax.text(10, 30, str_map, color='red')
        fig.savefig(f'output/detector_analysis/images/{str(count).zfill(6)}.jpg',
                    bbox_inches='tight')
        plt.close()

        if count == 4:
            with bz2.BZ2File(f'{img_addr}.pbz2', 'rb') as fp:
                fron = cPickle.load(fp)

            rgb_img = fron['rgb']
            sseg_img = fron['sseg']
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
                    # print(f'num_ins = {num_ins}')
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
                        print(f'cat_name = {lvis_cat_name}')
                        if x2 - x1 > 5 and y2 - y1 > 5:
                            bbox_list.append([x1, y1, x2, y2, lvis_cat_name])

            # plot the bboxes
            fig, ax = plt.subplots()
            ax.imshow(rgb_img)
            for bbox in bbox_list:
                x1, y1, x2, y2, cat_name = bbox
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, cat_name, color='green')
            plt.show()

            assert 1 == 2

        count += 1
