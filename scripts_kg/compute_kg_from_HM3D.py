'''
compute binary kg for HM3D train and val scenes
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

# =================== read the semantic map ===============
hm3d_to_lvis_dict = np.load(
    'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

# load the LVIS categories
with bz2.BZ2File('output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
    LVIS_dict = cPickle.load(fp)

# read the scene folders
scene_list = sorted(next(os.walk(f'{data_folder}/{split}'))[1])

num_classes = 310


for scene_floor in scene_list:
    print(f'scene_floor = {scene_floor}')
    scene_name, floor_id = scene_floor.split('_')

    cooccurrence_matrix = np.zeros((num_classes, num_classes))

    sample_name_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(
        glob.glob(f'{data_folder}/{split}/{scene_name}_{floor_id}/*.pbz2'))]

    for sample_name in sample_name_list:
        with bz2.BZ2File(f'{data_folder}/{split}/{scene_name}_{floor_id}/{sample_name}.pbz2', 'rb') as fp:
            fron = cPickle.load(fp)

        mat_dist = fron['mat_dist_to_cat'][0]
        mask = mat_dist >= 1

        idx_pos = list(np.argwhere(mask)[:, 0])

        for i in range(len(idx_pos) - 1):
            for j in range(i + 1, len(idx_pos)):
                a = idx_pos[i]
                b = idx_pos[j]
                cooccurrence_matrix[a, b] += 1
                cooccurrence_matrix[b, a] += 1

    np.save(
        f'output/kg_per_scene/co_matrix_{scene_name}_{floor_id}.npy', cooccurrence_matrix)
