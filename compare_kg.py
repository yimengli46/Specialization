'''
compare knowledge graphs with Jaccard similarity
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


df = pd.DataFrame(
    columns=['environment A', 'environment B', 'Jaccard Similarity'])
df['environment A'] = df['environment A'].astype(str)
df['environment B'] = df['environment B'].astype(str)
df['Jaccard Similarity'] = df['Jaccard Similarity'].astype(float)


with bz2.BZ2File('output/knowledge_graph/LVIS_relationships.pbz2', 'rb') as fp:
    LVIS_relationships = cPickle.load(fp)
    kg_VG = LVIS_relationships['adjacency_cat_id']

co_matrix_HM3D = np.load(
    'output/co_matrix_all_train_scene.npy', allow_pickle=True)

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
print(f'sim between VG and HM3D val = {sim}')

df = df.append(
    {
        'environment A': 'VisualGenome',
        'environment B': 'HM3D_train',
        'Jaccard Similarity': sim,
    },
    ignore_index=True)

# load the validation scenes
val_scene_co_matrix_folder = 'output/kg_per_scene'
comatrix_name_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(
    glob.glob(f'{val_scene_co_matrix_folder}/*.npy'))]

count = 0
count_HM3D_better = 0
for comatrix_name in comatrix_name_list:
    co_matrix_val = np.load(
        f'{val_scene_co_matrix_folder}/{comatrix_name}.npy', allow_pickle=True)

    adjacency_mat_HM3D_val = np.argwhere(co_matrix_val.flatten() > 0).flatten()

    sim_VG_and_HM3Dval = compute_jaccard_similarity(
        set(adjacency_mat_VG), set(adjacency_mat_HM3D_val))
    print(f'sim between VG and {comatrix_name} = {sim_VG_and_HM3Dval}')

    df = df.append(
        {
            'environment A': 'VisualGenome',
            'environment B': f'HM3D_val_{comatrix_name[10:]}',
            'Jaccard Similarity': sim_VG_and_HM3Dval,
        },
        ignore_index=True)

    sim_HM3Dtrain_and_HM3Dval = compute_jaccard_similarity(
        set(adjacency_mat_HM3D_train), set(adjacency_mat_HM3D_val))
    print(
        f'sim between HM3Dtrain and {comatrix_name} = {sim_HM3Dtrain_and_HM3Dval}')

    df = df.append(
        {
            'environment A': 'HM3d_train',
            'environment B': f'HM3D_val_{comatrix_name[10:]}',
            'Jaccard Similarity': sim_HM3Dtrain_and_HM3Dval,
        },
        ignore_index=True)

    count += 1
    if sim_HM3Dtrain_and_HM3Dval > sim_VG_and_HM3Dval:
        count_HM3D_better += 1

print(f'count = {count}, count_HM3D_better = {count_HM3D_better}')

html = df.to_html(float_format=lambda x: '%.4f' % x)
# write html to file
html_f = open(f'{val_scene_co_matrix_folder}/Jaccard.html', "w")
html_f.write(f'<h5>Knowledge Graph Similarity</h5>')
html_f.write(html)
html_f.close()
