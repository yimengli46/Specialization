'''
compute weighted kg between room and objs from CLIP predictions
'''

import numpy as np
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle
import os
import glob

# for each scene, load the image file
# get the distance vector
# add to the co-occurrence matrix
split = 'val'  # 'train'
data_folder = 'output/training_data_input_view_by_densely_sample_locations'
clip_folder = 'output/CLIP_room_type'

output_folder = 'output/weighted_kg/weighted_kg_room_and_obj'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

saved_folder = f'{output_folder}/{split}'
if not os.path.exists(saved_folder):
    os.mkdir(saved_folder)

# =================== read the semantic map ===============
hm3d_to_lvis_dict = np.load(
    'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

# load the LVIS categories
with bz2.BZ2File('output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
    LVIS_dict = cPickle.load(fp)

# read the scene folders
scene_list = sorted(next(os.walk(f'{data_folder}/{split}'))[1])

num_classes = 310
num_rooms = 10 + 1  # 10 room types + 'unknown'
thresh_room_conf = 0.5

for scene_floor in scene_list:
    print(f'scene_floor = {scene_floor}')
    scene_name, floor_id = scene_floor.split('_')

    sample_name_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(
        glob.glob(f'{data_folder}/{split}/{scene_name}_{floor_id}/*.pbz2'))]

    cooccurrence_matrix = np.ones(
        (num_rooms, num_classes, len(sample_name_list)), dtype=np.int16) * -1

    for i_sample, sample_name in enumerate(sample_name_list):
        # load the obj detections
        with bz2.BZ2File(f'{data_folder}/{split}/{scene_name}_{floor_id}/{sample_name}.pbz2', 'rb') as fp:
            fron = cPickle.load(fp)
        # load the room predictions
        clip_npy = np.load(
            f'{clip_folder}/{split}/{scene_name}_{floor_id}/{sample_name}_clip_room_types.npy', allow_pickle=True)

        mat_dist = fron['map_dist_to_cat'][0]
        mask = (mat_dist == 1)

        idx_pos = list(np.argwhere(mask)[:, 0])

        # get room type
        idx_room = np.argmax(clip_npy)
        if clip_npy[idx_room] < thresh_room_conf:
            idx_room = num_rooms - 1

        # for j in range(len(idx_pos)):
        #    b = idx_pos[j]
        cooccurrence_matrix[idx_room, idx_pos, i_sample] = 1

        mask_negative = cooccurrence_matrix[idx_room, :, i_sample] == -1
        cooccurrence_matrix[idx_room, mask_negative, i_sample] = 0

        # assert 1 == 2

    np.save(
        f'{saved_folder}/co_matrix_{scene_name}_{floor_id}.npy', cooccurrence_matrix)
