'''
compare weighted knowledge graphs with Cosine similarity between room types and objs
The weighted knowledge graph is computed similarly as Stuart Russel's paper.
'''

import numpy as np
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle
import os
import glob
import pandas as pd
import random


def compute_mle(k_success, n_trials):
    mle_estimate = k_success / (n_trials + 1e-10)
    return mle_estimate


def compute_weighted_jaccard_similarity(set1, set2):
    assert set1.shape[0] == 1
    assert set2.shape[0] == 1
    assert len(set1.shape) == 2
    assert len(set2.shape) == 2
    n1 = set1.shape[1]
    n2 = set2.shape[1]
    assert n1 == n2
    union = np.concatenate((set1, set2), axis=0)
    min_union = union.min(axis=0)
    max_union = union.max(axis=0)
    weighted_jaccard_similarity = min_union.sum() / (max_union.sum() + 1e-10)
    return weighted_jaccard_similarity


def cosine_similarity(set1, set2):
    set1 = set1[0]
    set2 = set2[0]
    dot_product = np.sum(set1 * set2)
    norm1 = np.linalg.norm(set1)
    norm2 = np.linalg.norm(set2)
    sim = dot_product / (norm1 * norm2)
    return sim


num_rooms = 10 + 1  # 10 room types + 'unknown'
num_classes = 310

room_types = ['a living room', 'a bathroom', 'a dining room', 'a kitchen',
              'a bedroom', 'a pantry', 'an office', 'a garage', 'outdoor', 'a corridor', 'unknown']

df = pd.DataFrame(
    columns=['environment A', 'environment B', 'Cosine Similarity'])
df['environment A'] = df['environment A'].astype(str)
df['environment B'] = df['environment B'].astype(str)
df['Cosine Similarity'] = df['Cosine Similarity'].astype(
    float)

# ================== compute weighted co-occurrence matrix for HM3D train scenes
'''
# load the HM3D train scenes
train_scene_co_matrix_folder = 'output/weighted_kg/weighted_kg_room_and_obj/train'
comatrix_name_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(
    glob.glob(f'{train_scene_co_matrix_folder}/*.npy'))]

# first dim count number of 0, second dim count number of 1
z_ij_train = np.zeros((num_rooms, num_classes, 2), dtype=np.int32)

for comatrix_name in comatrix_name_list:
    print(f'comatrix_name = {comatrix_name}')
    co_matrix = np.load(
        f'{train_scene_co_matrix_folder}/{comatrix_name}.npy', allow_pickle=True)

    num_images = co_matrix.shape[2]
    for i_image in range(num_images):
        idx_zero = np.where(co_matrix[:, :, i_image] == 0)
        idx_one = np.where(co_matrix[:, :, i_image] == 1)

        rs, cs = idx_zero
        for i_r in range(len(rs)):
            z_ij_train[rs[i_r], cs[i_r], 0] += 1

        rs, cs = idx_one
        for i_r in range(len(rs)):
            z_ij_train[rs[i_r], cs[i_r], 1] += 1

print('estimate weighted co-occurrence matrix via MLE ...')
weighted_co_matrix = np.zeros((num_rooms, num_classes), dtype=np.float32)

for r in range(num_rooms):
    for c in range(num_classes):
        k_success = z_ij_train[r, c, 1]  # equals to number of 1s
        n_trials = z_ij_train[r, c, 0] + z_ij_train[r, c, 1]
        mle_estimate = compute_mle(k_success, n_trials)
        weighted_co_matrix[r, c] = mle_estimate

np.save(f'output/weighted_kg/z_ij_room_and_obj_train_all.npy', z_ij_train)
np.save(f'output/weighted_kg/weighted_co_matrix_room_and_obj_train_all.npy', weighted_co_matrix)

assert 1 == 2
'''

z_ij_train = np.load(
    'output/weighted_kg/z_ij_room_and_obj_train_all.npy', allow_pickle=True)
weighted_co_matrix_train = np.load(
    'output/weighted_kg/weighted_co_matrix_room_and_obj_train_all.npy', allow_pickle=True)

'''
# === for visualize the probability graph between room types and other objects
row_sum = weighted_co_matrix_train.sum(axis=1)
maximum_row_idx = 1
maximum_row = weighted_co_matrix_train[maximum_row_idx]

hm3d_to_lvis_dict = np.load(
    'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

# load the LVIS categories
with bz2.BZ2File(f'output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
    LVIS_dict = cPickle.load(fp)

goal_obj_list = sorted(list(set(hm3d_to_lvis_dict.values())))  # size: 351

goal_obj_index_list = list(set(LVIS_dict['rowid2catid_dict'][LVIS_dict['cat_synonyms'].index(
    cat_syn)] for cat_syn in goal_obj_list))  # size: 310

lvis_cat_name_to_lvis_id_dict = {cat_syn: LVIS_dict['rowid2catid_dict'][LVIS_dict['cat_synonyms'].index(
    cat_syn)] for cat_syn in goal_obj_list}

lvis_id_to_lvis_cat_names_dict = {}
for cat_syn in goal_obj_list:
    lvis_id = LVIS_dict['rowid2catid_dict'][LVIS_dict['cat_synonyms'].index(
        cat_syn)]
    if lvis_id in lvis_id_to_lvis_cat_names_dict:
        lvis_id_to_lvis_cat_names_dict[lvis_id].append(cat_syn)
    else:
        lvis_id_to_lvis_cat_names_dict[lvis_id] = [cat_syn]

my_list = sorted(lvis_id_to_lvis_cat_names_dict.keys())
print(f'target = {lvis_id_to_lvis_cat_names_dict[my_list[maximum_row_idx]]}')
for idx, lvis_id in enumerate(sorted(lvis_id_to_lvis_cat_names_dict.keys())):
    print(
        f'idx = {idx}, obj = {lvis_id_to_lvis_cat_names_dict[lvis_id]}, prob = {maximum_row[idx]}')

# assert 1 == 2
'''


# ======================== compare with HM3D val entirely ========================
# load the HM3D val scenes
val_scene_co_matrix_folder = 'output/weighted_kg/weighted_kg_room_and_obj/val'
comatrix_name_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(
    glob.glob(f'{val_scene_co_matrix_folder}/*.npy'))]

# first dim count number of 0, second dim count number of 1
z_ij_val = np.zeros((num_rooms, num_classes, 2), dtype=np.int32)

for comatrix_name in comatrix_name_list:
    print(f'comatrix_name = {comatrix_name}')
    co_matrix = np.load(
        f'{val_scene_co_matrix_folder}/{comatrix_name}.npy', allow_pickle=True)

    num_images = co_matrix.shape[2]
    for i_image in range(num_images):
        idx_zero = np.where(co_matrix[:, :, i_image] == 0)
        idx_one = np.where(co_matrix[:, :, i_image] == 1)

        rs, cs = idx_zero
        for i_r in range(len(rs)):
            z_ij_val[rs[i_r], cs[i_r], 0] += 1

        rs, cs = idx_one
        for i_r in range(len(rs)):
            z_ij_val[rs[i_r], cs[i_r], 1] += 1

print('estimate weighted co-occurrence matrix via MLE ...')
weighted_co_matrix_val = np.zeros((num_rooms, num_classes), dtype=np.float32)

for r in range(num_rooms):
    for c in range(num_classes):
        k_success = z_ij_val[r, c, 1]  # equals to number of 1s
        n_trials = z_ij_val[r, c, 0] + z_ij_val[r, c, 1]
        mle_estimate = compute_mle(k_success, n_trials)
        weighted_co_matrix_val[r, c] = mle_estimate

weighted_sim = cosine_similarity(
    weighted_co_matrix_train.reshape(1, -1), weighted_co_matrix_val.reshape(1, -1))

df = df.append(
    {
        'environment A': 'HM3d_train_all',
        'environment B': f'HM3D_val_all',
        'Cosine Similarity': weighted_sim,
    },
    ignore_index=True)

# ======================== compare with HM3D val each scene separately ========================
HM3D_val_all_sim_list = []

HM3D_val_portion_5_sim_list = []
HM3D_val_portion_25_sim_list = []
HM3D_val_portion_50_sim_list = []

portion_list = [.05, .25, .5]

for comatrix_name in comatrix_name_list:
    print(f'comatrix_name = {comatrix_name}')
    co_matrix = np.load(
        f'{val_scene_co_matrix_folder}/{comatrix_name}.npy', allow_pickle=True)

    # first dim count number of 0, second dim count number of 1
    z_ij_val = np.zeros((num_rooms, num_classes, 2), dtype=np.int32)

    num_images = co_matrix.shape[2]
    for i_image in range(num_images):
        idx_zero = np.where(co_matrix[:, :, i_image] == 0)
        idx_one = np.where(co_matrix[:, :, i_image] == 1)

        rs, cs = idx_zero
        for i_r in range(len(rs)):
            z_ij_val[rs[i_r], cs[i_r], 0] += 1

        rs, cs = idx_one
        for i_r in range(len(rs)):
            z_ij_val[rs[i_r], cs[i_r], 1] += 1

    print('estimate weighted co-occurrence matrix via MLE ...')
    weighted_co_matrix_val = np.zeros(
        (num_rooms, num_classes), dtype=np.float32)

    for r in range(num_rooms):
        for c in range(num_classes):
            k_success = z_ij_val[r, c, 1]  # equals to number of 1s
            n_trials = z_ij_val[r, c, 0] + z_ij_val[r, c, 1]
            mle_estimate = compute_mle(k_success, n_trials)
            weighted_co_matrix_val[r, c] = mle_estimate

    # =============== compute jaccard on val scene objects only
    # find object rows to ignore
    if True:
        rows_not_ignore = set(np.unique(np.where(z_ij_val[:, :, 1] > 0)[0]))
        rows_ignore = set(np.array(range(num_rooms))) - rows_not_ignore
        # zero out VG and HM3D train kg ignored rows
        weighted_co_matrix_train_clone = weighted_co_matrix_train.copy()
        for r in rows_ignore:
            weighted_co_matrix_train_clone[r, :] = 0

    weighted_sim = cosine_similarity(
        weighted_co_matrix_train_clone.reshape(1, -1), weighted_co_matrix_val.reshape(1, -1))

    df = df.append(
        {
            'environment A': 'HM3d_train_all',
            'environment B': f'HM3D_val_{comatrix_name[10:]}_0',
            'Cosine Similarity': weighted_sim,
        },
        ignore_index=True)

    HM3D_val_all_sim_list.append(weighted_sim)

    # ===== compare with HM3D val each scene but with a small portion of val data ========================
    if True:
        for portion in portion_list:
            # first dim count number of 0, second dim count number of 1
            z_ij_val = np.zeros((num_rooms, num_classes, 2), dtype=np.int32)

            num_images = co_matrix.shape[2]
            sampled_img_id_list = np.random.choice(
                num_images, int(portion * num_images), replace=False)
            for i_image in sampled_img_id_list:
                idx_zero = np.where(co_matrix[:, :, i_image] == 0)
                idx_one = np.where(co_matrix[:, :, i_image] == 1)

                rs, cs = idx_zero
                for i_r in range(len(rs)):
                    z_ij_val[rs[i_r], cs[i_r], 0] += 1

                rs, cs = idx_one
                for i_r in range(len(rs)):
                    z_ij_val[rs[i_r], cs[i_r], 1] += 1

            print('estimate weighted co-occurrence matrix via MLE ...')
            weighted_co_matrix_train_val_portion = np.zeros(
                (num_rooms, num_classes), dtype=np.float32)

            for r in range(num_rooms):
                for c in range(num_classes):
                    k_success = z_ij_val[r, c, 1]  # equals to number of 1s
                    n_trials = z_ij_val[r, c, 0] + z_ij_val[r, c, 1]
                    alpha = z_ij_train[r, c, 1] + 1
                    beta = z_ij_train[r, c, 0] + 1
                    k_success = k_success * \
                        (alpha + beta - 2) / (n_trials + 1e-10)
                    n_trials = alpha + beta - 2
                    MAP_estimate = (k_success + alpha - 1) / \
                        (n_trials + alpha - 1 + beta - 1 + 1e-10)
                    weighted_co_matrix_train_val_portion[r, c] = MAP_estimate

            # =============== compute jaccard on val scene objects only
            # find object rows to ignore
            if True:
                weighted_co_matrix_train_val_portion_clone = weighted_co_matrix_train_val_portion.copy()
                # zero out HM3D train kg ignored rows
                for r in rows_ignore:
                    weighted_co_matrix_train_val_portion_clone[r, :] = 0

            weighted_sim_portion = cosine_similarity(
                weighted_co_matrix_train_val_portion_clone.reshape(1, -1), weighted_co_matrix_val.reshape(1, -1))

            df = df.append(
                {
                    'environment A': 'HM3d_train_all',
                    'environment B': f'HM3D_val_{comatrix_name[10:]}_{portion}',
                    'Cosine Similarity': weighted_sim_portion,
                },
                ignore_index=True)

            if portion == .05:
                HM3D_val_portion_5_sim_list.append(weighted_sim_portion)
            elif portion == .25:
                HM3D_val_portion_25_sim_list.append(weighted_sim_portion)
            elif portion == .5:
                HM3D_val_portion_50_sim_list.append(weighted_sim_portion)


html = df.to_html(float_format=lambda x: '%.4f' % x)
# write html to file
html_f = open(f'output/weighted_kg/Cosine_similarity_room_and_obj.html', "w")
html_f.write(f'<h5>Knowledge Graph Similarity</h5>')
html_f.write(html)

# ================== write mean ================
df2 = pd.DataFrame(
    columns=['environment A', 'environment B', 'Mean Cosine Similarity'])
df2['environment A'] = df2['environment A'].astype(str)
df2['environment B'] = df2['environment B'].astype(str)
df2['Mean Cosine Similarity'] = df2['Mean Cosine Similarity'].astype(float)

df2 = df2.append(
    {
        'environment A': 'HM3D_train_val_0%',
        'environment B': f'HM3D_val_each_scene',
        'Mean Cosine Similarity': np.array(HM3D_val_all_sim_list).mean(),
    },
    ignore_index=True)

df2 = df2.append(
    {
        'environment A': 'HM3D_train_val_5%',
        'environment B': f'HM3D_val_each_scene',
        'Mean Cosine Similarity': np.array(HM3D_val_portion_5_sim_list).mean(),
    },
    ignore_index=True)

df2 = df2.append(
    {
        'environment A': 'HM3D_train_val_25%',
        'environment B': f'HM3D_val_each_scene',
        'Mean Cosine Similarity': np.array(HM3D_val_portion_25_sim_list).mean(),
    },
    ignore_index=True)

df2 = df2.append(
    {
        'environment A': 'HM3D_train_val_50%',
        'environment B': f'HM3D_val_each_scene',
        'Mean Cosine Similarity': np.array(HM3D_val_portion_50_sim_list).mean(),
    },
    ignore_index=True)

html = df2.to_html(float_format=lambda x: '%.4f' % x)
html_f.write(f'<h5>Mean over all val scenes</h5>')
html_f.write(html)

html_f.close()
