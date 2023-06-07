import numpy as np
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle
import os
import glob
import pandas as pd
import random
import matplotlib.patches as patches
from dataloader_input_view_by_densely_sampled_locations import process_lvis_dict

split = 'val'
img_folder = 'output/training_data_input_view_by_densely_sample_locations_FULL_SIZE'
clip_folder = '/home/yimeng/ARGO_scratch/topo_map_specialization/Specialization/output/CLIP_room_type/val'
saved_folder = 'output/vis_rules'

hm3d_to_lvis_dict = np.load(
    'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

# load the LVIS categories
with bz2.BZ2File(f'output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
    LVIS_dict = cPickle.load(fp)
    lvis_cat_synonyms_list = LVIS_dict['cat_synonyms']
    lvis_rowid_to_catid_dict = LVIS_dict['rowid2catid_dict']
    lvis_cat_synonyms_embedding = LVIS_dict['cat_synonyms_embedding']
    lvis_cat_index_embedding = LVIS_dict['cat_id_embedding']

goal_obj_list, goal_obj_index_list, lvis_cat_name_to_lvis_id_dict, lvis_id_to_lvis_cat_names_dict = process_lvis_dict(
    hm3d_to_lvis_dict, LVIS_dict)

# ================ define the objects, rooms and targets ===========================
room_types = ['a living room', 'a bathroom', 'a dining room', 'a kitchen',
              'a bedroom', 'a pantry', 'an office', 'a garage', 'outdoor', 'a corridor', 'unknown']

common_objs = [19, 232, 464, 558, 889, 285, 367, 610, 831, 1050, 181, 438, 351, 982, 77, 170, 390,
               135, 837, 1139, 609, 961, 1097, 468, 710, 1013, 1018, 1108, 1077, 957, 955, 68, 444, 677]
common_objs_row_ids = [goal_obj_index_list.index(i) for i in common_objs]


# targets = set([19, 232, 464, 558, 889, 982, 135, 837, 77, 170, 1097, 1077])

targets = {
    'chair': [19, 232, 464, 558, 889],
    'sofa': [982],
    'plant': [135, 837, 1139],
    'bed': [77, 170],
    'toilet': [1097],
    'tv': [1077],
}

# ==================== load the pre-computed co-occurrences ===========================
weighted_co_matrix_obj_and_obj = np.load(
    'output/weighted_kg/weighted_co_matrix_obj_and_obj_train_all.npy', allow_pickle=True)
weighted_co_matrix_room_and_obj = np.load(
    'output/weighted_kg/weighted_co_matrix_room_and_obj_train_all.npy', allow_pickle=True)

scene_list = sorted(next(os.walk(f'{img_folder}/{split}'))[1])

for scene_name in scene_list:

    # scene_name = '00800-TEEsavR23oF_0'
    if not os.path.exists(f'{saved_folder}/{scene_name}'):
        os.mkdir(f'{saved_folder}/{scene_name}')

    # read the img
    sample_name_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(
        glob.glob(f'{img_folder}/{split}/{scene_name}/*.pbz2'))]
    sample_name_list = random.choices(sample_name_list, k=100)

    for sample_name in sample_name_list:
        with bz2.BZ2File(f'{img_folder}/{split}/{scene_name}/{sample_name}.pbz2', 'rb') as fp:
            fron = cPickle.load(fp)

            rgb_img = fron['rgb']
            bbox_list = fron['bbox']
            map_dist_to_cat = fron['map_dist_to_cat']

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
            ax[0].imshow(rgb_img)
            # draw the object detector bboxes
            for bbox in bbox_list:
                x1, y1, x2, y2, cat_name = bbox
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                ax[0].add_patch(rect)
                ax[0].text(x1, y1, cat_name, color='green')
            ax[0].get_xaxis().set_visible(False)
            ax[0].get_yaxis().set_visible(False)
            ax[0].set_title('egocentric view')

            # check the common objects in the view
            view_common_obj_row_ids = set(np.where(map_dist_to_cat == 1)[1]).intersection(common_objs_row_ids)

            text_x = 0.05  # Adjust the x-coordinate based on image width
            text_y = 0.98  # Adjust the y-coordinate based on image height
            count_y = 0

            for goal_obj in list(targets.keys()):
                # randomly pick a target object
                # goal_obj = random.choice(list(targets.keys()))
                print(f'goal_obj = {goal_obj}')
                goal_obj_row_ids = [goal_obj_index_list.index(i) for i in targets[goal_obj]]

                text = f'target: {goal_obj}'
                ax[1].text(text_x, text_y - (count_y * 0.02), text, fontsize=12)
                count_y += 1

                for common_obj_row_id in view_common_obj_row_ids:
                    cooccur_value = weighted_co_matrix_obj_and_obj[goal_obj_row_ids, common_obj_row_id].max()
                    common_obj_name = lvis_id_to_lvis_cat_names_dict[goal_obj_index_list[common_obj_row_id]]
                    text = f'{common_obj_name}: {cooccur_value:.3f}'
                    ax[1].text(text_x * 2, text_y - (count_y * .02), text, fontsize=12)
                    count_y += 1

                # load the room types
                room_type_preds = np.load(
                    f'{clip_folder}/{scene_name}/{sample_name}_clip_room_types.npy', allow_pickle=True)
                # get room type
                idx_room = np.argmax(room_type_preds)
                if room_type_preds[idx_room] < 0.5:
                    idx_room = len(room_types) - 1

                cooccur_value = weighted_co_matrix_room_and_obj[idx_room, goal_obj_row_ids].max()
                room_name = room_types[idx_room]
                text = f'{room_name}: {cooccur_value:.3f}'
                ax[1].text(text_x * 2, text_y - (count_y * .02), text, fontsize=12)
                count_y += 1

            # plt.show()
            fig.savefig(f'{saved_folder}/{scene_name}/{sample_name}.jpg',
                        bbox_inches='tight')
            plt.close()
