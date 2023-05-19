'''
compare binary knowledge graphs with Jaccard similarity
'''

import numpy as np
import matplotlib.pyplot as plt
import bz2
import _pickle as cPickle
import os
import glob
import pandas as pd

# load the VG cooccurence matrix
# load the HM3D matrix
# normalize the the graph edge weights
# compare graph similarity


def compute_jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    # print(f'intersection = {intersection}')
    union = set1.union(set2)
    jaccard_similarity = len(intersection) / len(union)
    return jaccard_similarity


def cosine_similarity(set1, set2):
    dot_product = np.sum(set1 * set2)
    norm1 = np.linalg.norm(set1)
    norm2 = np.linalg.norm(set2)
    sim = dot_product / (norm1 * norm2)
    return sim


df = pd.DataFrame(
    columns=['environment A', 'environment B', 'Jaccard Similarity', 'Cosine Similarity'])
df['environment A'] = df['environment A'].astype(str)
df['environment B'] = df['environment B'].astype(str)
df['Jaccard Similarity'] = df['Jaccard Similarity'].astype(float)
df['Cosine Similarity'] = df['Cosine Similarity'].astype(float)


with bz2.BZ2File('output/knowledge_graph/LVIS_relationships.pbz2', 'rb') as fp:
    LVIS_relationships = cPickle.load(fp)
    kg_VG = LVIS_relationships['adjacency_cat_id']

co_matrix_HM3D = np.load(
    'output/kg_per_scene/co_matrix_all_train_scene.npy', allow_pickle=True)

# normalize HM3D kg
thresh = 10
if False:
    edge_weights = co_matrix_HM3D.flatten()
    edge_weights = edge_weights[edge_weights > thresh]
    plt.hist(edge_weights, bins=list(range(0, 100, 5)))
    plt.show()

binary_co_matrix_HM3D = co_matrix_HM3D > thresh

adjacency_mat_VG = np.argwhere(kg_VG.flatten() == 1).flatten()
adjacency_mat_HM3D_train = np.argwhere(
    binary_co_matrix_HM3D.flatten()).flatten()

sim = compute_jaccard_similarity(
    set(adjacency_mat_VG), set(adjacency_mat_HM3D_train))
cosine_sim = cosine_similarity(
    kg_VG.flatten(), binary_co_matrix_HM3D.flatten())
print(f'sim between VG and HM3D val = {sim}')

df = df.append(
    {
        'environment A': 'VisualGenome',
        'environment B': 'HM3D_train',
        'Jaccard Similarity': sim,
        'Cosine Similarity': cosine_sim
    },
    ignore_index=True)

# load the validation scenes
val_scene_co_matrix_folder = 'output/kg_per_scene'
comatrix_name_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(
    glob.glob(f'{val_scene_co_matrix_folder}/*.npy'))]
comatrix_name_list.remove('co_matrix_all_train_scene')

# ====================== compare with HM3D val scene entirely ===================
co_matrix_val_all = np.zeros((310, 310))
for comatrix_name in comatrix_name_list:
    co_matrix_val = np.load(
        f'{val_scene_co_matrix_folder}/{comatrix_name}.npy', allow_pickle=True)
    co_matrix_val_all += co_matrix_val

# continue comparison
adjacency_mat_HM3D_val = np.argwhere(co_matrix_val_all.flatten() > 0).flatten()

sim_VG_and_HM3Dval = compute_jaccard_similarity(
    set(adjacency_mat_VG), set(adjacency_mat_HM3D_val))
cos_sim_VG_and_HM3Dval = cosine_similarity(
    kg_VG.flatten(), co_matrix_val_all.flatten())
print(f'sim between VG and {comatrix_name} = {sim_VG_and_HM3Dval}')

df = df.append(
    {
        'environment A': 'VisualGenome',
        'environment B': f'HM3D_val_all',
        'Jaccard Similarity': sim_VG_and_HM3Dval,
        'Cosine Similarity': cos_sim_VG_and_HM3Dval
    },
    ignore_index=True)

sim_HM3Dtrain_and_HM3Dval = compute_jaccard_similarity(
    set(adjacency_mat_HM3D_train), set(adjacency_mat_HM3D_val))
cos_sim_HM3Dtrain_and_HM3Dval = cosine_similarity(
    binary_co_matrix_HM3D.flatten(), co_matrix_val_all.flatten())
print(
    f'sim between HM3Dtrain and {comatrix_name} = {sim_HM3Dtrain_and_HM3Dval}')

df = df.append(
    {
        'environment A': 'HM3d_train',
        'environment B': f'HM3D_val_all',
        'Jaccard Similarity': sim_HM3Dtrain_and_HM3Dval,
        'Cosine Similarity': cos_sim_HM3Dtrain_and_HM3Dval
    },
    ignore_index=True)

# ================ compare with each HM3D val scene ==================================
count = 0
count_HM3D_better = 0
VG_sim_list = []
HM3D_train_sim_list = []
VG_cos_sim_list = []
HM3D_train_cos_sim_list = []
for comatrix_name in comatrix_name_list:
    co_matrix_val = np.load(
        f'{val_scene_co_matrix_folder}/{comatrix_name}.npy', allow_pickle=True)

    # =============== compute jaccard on val scene objects only
    # find object rows to ignore
    if True:
        rows_not_ignore = set(np.unique(np.where(co_matrix_val == 1)[0]))
        rows_ignore = set(np.array(range(310))) - rows_not_ignore
        # zero out VG and HM3D train kg ignored rows
        kg_VG_clone = kg_VG.copy()
        for r in rows_ignore:
            kg_VG_clone[r, :] = 0
            kg_VG_clone[:, r] = 0
        adjacency_mat_VG = np.argwhere(kg_VG_clone.flatten() == 1).flatten()

        binary_co_matrix_HM3D_clone = binary_co_matrix_HM3D.copy()
        for r in rows_ignore:
            binary_co_matrix_HM3D_clone[r, :] = 0
            binary_co_matrix_HM3D_clone[:, r] = 0
        adjacency_mat_HM3D_train = np.argwhere(
            binary_co_matrix_HM3D_clone.flatten()).flatten()

    # continue comparison
    adjacency_mat_HM3D_val = np.argwhere(co_matrix_val.flatten() > 0).flatten()

    sim_VG_and_HM3Dval = compute_jaccard_similarity(
        set(adjacency_mat_VG), set(adjacency_mat_HM3D_val))
    cos_sim_VG_and_HM3Dval = cosine_similarity(
        kg_VG_clone.flatten(), co_matrix_val.flatten())
    print(f'sim between VG and {comatrix_name} = {sim_VG_and_HM3Dval}')

    df = df.append(
        {
            'environment A': 'VisualGenome',
            'environment B': f'HM3D_val_{comatrix_name[10:]}',
            'Jaccard Similarity': sim_VG_and_HM3Dval,
            'Cosine Similarity': cos_sim_VG_and_HM3Dval
        },
        ignore_index=True)

    sim_HM3Dtrain_and_HM3Dval = compute_jaccard_similarity(
        set(adjacency_mat_HM3D_train), set(adjacency_mat_HM3D_val))
    cos_sim_HM3Dtrain_and_HM3Dval = cosine_similarity(
        binary_co_matrix_HM3D_clone.flatten(), co_matrix_val.flatten())
    print(
        f'sim between HM3Dtrain and {comatrix_name} = {sim_HM3Dtrain_and_HM3Dval}')

    df = df.append(
        {
            'environment A': 'HM3D_train',
            'environment B': f'HM3D_val_{comatrix_name[10:]}',
            'Jaccard Similarity': sim_HM3Dtrain_and_HM3Dval,
            'Cosine Similarity': cos_sim_HM3Dtrain_and_HM3Dval
        },
        ignore_index=True)

    VG_sim_list.append(sim_VG_and_HM3Dval)
    HM3D_train_sim_list.append(sim_HM3Dtrain_and_HM3Dval)
    VG_cos_sim_list.append(cos_sim_VG_and_HM3Dval)
    HM3D_train_cos_sim_list.append(cos_sim_HM3Dtrain_and_HM3Dval)

    count += 1
    if sim_HM3Dtrain_and_HM3Dval > sim_VG_and_HM3Dval:
        count_HM3D_better += 1

print(f'count = {count}, count_HM3D_better = {count_HM3D_better}')

html = df.to_html(float_format=lambda x: '%.4f' % x)
# write html to file
html_f = open(
    f'{val_scene_co_matrix_folder}/Jaccard_Cosine_val_objs_only.html', "w")
html_f.write(f'<h5>Knowledge Graph Similarity</h5>')
html_f.write(html)

# compute mean
df2 = pd.DataFrame(
    columns=['environment A', 'environment B', 'Mean Jaccard Similarity', 'Mean Cosine Similarity'])
df2['environment A'] = df2['environment A'].astype(str)
df2['environment B'] = df2['environment B'].astype(str)
df2['Mean Jaccard Similarity'] = df2['Mean Jaccard Similarity'].astype(float)
df2['Mean Cosine Similarity'] = df2['Mean Cosine Similarity'].astype(float)

df2 = df2.append(
    {
        'environment A': 'VisualGenome',
        'environment B': f'HM3D_val_each_scene',
        'Mean Jaccard Similarity': np.array(VG_sim_list).mean(),
        'Mean Cosine Similarity': np.array(VG_cos_sim_list).mean()
    },
    ignore_index=True)

df2 = df2.append(
    {
        'environment A': 'HM3D_train',
        'environment B': f'HM3D_val_each_scene',
        'Mean Jaccard Similarity': np.array(HM3D_train_sim_list).mean(),
        'Mean Cosine Similarity': np.array(HM3D_train_cos_sim_list).mean()
    },
    ignore_index=True)

html = df2.to_html(float_format=lambda x: '%.4f' % x)
html_f.write(f'<h5>Mean over all val scenes</h5>')
html_f.write(html)

html_f.close()
