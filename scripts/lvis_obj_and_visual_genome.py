'''
Find the connection between lvis categories and VG
'''
import json
import os
from collections import Counter
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import bz2
import _pickle as cPickle


def get_triplet(relationship):
    try:
        subj_name = relationship['subject']['names'][0]
    except KeyError:
        subj_name = relationship['subject']['name']
    try:
        obj_name = relationship['object']['names'][0]
    except:
        obj_name = relationship['object']['name']
    return subj_name.lower(), relationship['predicate'].lower(), obj_name.lower()


def add_or_append(final_dict, rel, rel_objs):
    if rel not in final_dict.keys():
        final_dict[rel] = rel_objs
        return final_dict
    else:
        for subj, count in final_dict[rel].items():
            if (subj in rel_objs.keys()):
                count += rel_objs[subj]
            final_dict[rel][subj] = count
    return final_dict


def ensuredirs(fpath):
    fdir = osp.dirname(fpath)
    if not osp.exists(fdir):
        os.makedirs(fdir)
    return fpath


def find_synonyms_with_sentenceTransformer(model, obj_idx, obj, lvis_objs_embedding,
                                           vg_objs_embedding, vg_full_list, n=5, thresh=0.65):
    query_embedding = lvis_objs_embedding[obj_idx]
    similarity = util.dot_score(
        query_embedding, vg_objs_embedding).squeeze().numpy()
    idx_list_similarity = np.argsort(similarity)[::-1]
    result_list = []
    n = n if len(vg_full_list) >= n else len(vg_full_list)
    for idx in range(n):
        sim_idx = idx_list_similarity[idx]
        if similarity[sim_idx] >= thresh:
            result_list.append(vg_full_list[sim_idx])
    return result_list


# =============================== load VG data ===============================
output_folder = 'output/knowledge_graph'
vg_folder = '../MJOLNIR-master/kg_prep/kg_data'


print('load visual genome relationships ...')
# with open(f'{vg_folder}/small_relationships.json') as f:
with open(f'{vg_folder}/relationships.json') as f:
    data = json.load(f)


print("Filtering LVIS goal objects from VG...")
# =============================== extract the subjects from VG relationships ====================
'''
Analyze VG relationship json file.
extract subjects dict having (k, v) like:
'dishcloths': {'on': {'countertop': 1}}
'''
subjects = {}
for image in data:
    for relationship in image['relationships']:
        subj_name, rel_name, obj_name = get_triplet(relationship)
        # print(f'relationship = {relationship}')
        # print(
        #     f'subj = {subj_name}, rel_name = {rel_name}, obj_name = {obj_name}')
        # print(f'------------------------------------------------------------------')
        subj = subjects.get(subj_name, {})
        rel = subj.get(rel_name, {})
        rel[obj_name] = rel.get(obj_name, 0) + 1
        subj[rel_name] = rel
        subjects[subj_name] = subj

vg_full_list = sorted(list(subjects.keys()))

# load lvis goal objects
hm3d_to_lvis_dict = np.load(
    f'{output_folder}/hm3d_to_lvis_dict.npy', allow_pickle=True).item()
lvis_objs = sorted(list(set(hm3d_to_lvis_dict.values())))
#assert 1 == 2

# ==================== compute the embedding of lvis obj and vm obj ==================
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
lvis_objs_embedding = model.encode(lvis_objs)
vg_objs_embedding = model.encode(vg_full_list)


# ==========================compute the synonyms between lvis and vg===========================

lvis_vg_map = {}  # lvis to visual genome mapping
vg_list = []  # visual genome subjects got matched

# map the lvis objects to vg objects
for i, obj in enumerate(lvis_objs):
    a = lvis_objs[i]
    # b = difflib.get_close_matches(
    #     obj, possibilities=vg_full_list, n=5, cutoff=0.9)
    b = find_synonyms_with_sentenceTransformer(
        model, i, obj, lvis_objs_embedding, vg_objs_embedding, vg_full_list, n=5, thresh=0.65)
    #print(f'a = {a}, b = {b}')
    lvis_vg_map[a] = b
    vg_list.extend(lvis_vg_map[lvis_objs[i]])


filtered_subjects = {}
# only keep the vg objects having similar words in lvis goal objects
for subj_name, subject in subjects.items():
    if subj_name in vg_list:
        filtered_subjects[subj_name] = subject


# ================================================================================
print("Done filtering. Starting object pruning...")
# extract
all_objs = sorted(list(filtered_subjects.keys()))

# ====================clean up filtered subjects so rel_obj is also in vg_list=======
"""
filtered_subjects save (k, v) like:
'fishtank': {'with': {}, 'has': {'box': 1}} 
'printef': {'on': {'desk': 1}}}

obj = 'printer'
rel = 'on'
rel_objs = 'desk'
if desk is not in vg_list, it's removed.

So the block below removes relationships where rel_obj is not in vg_list.
"""
for obj in all_objs:

    data = filtered_subjects[obj]

    for rel, rel_objs in data.items():
        # check if the rel_obj is in the vg_list
        rel_objs = dict((k, v) for (k, v) in rel_objs.items() if k in vg_list)
        data[rel] = rel_objs
    data = {k: v for k, v in data.items() if len(v) > 0}
    if (len(data) == 0):
        filtered_subjects.pop(obj)


# ==========================================================================
"""
Count the number of times the relationship built between a lvis goal obj and a vg object.

lvis obj is linked to vg obj through lvis_vg_map.
filtered_subjects save (k, v) like (vg_obj1, {'on': {vg_obj2: number_of_times}})

then using links from lvis_vg_map,
final_rels save (k, v) like (lvis_obj, {'on': {vg_obj2: number_of_times}})
"""
all_noisy_objs = sorted(list(filtered_subjects.keys()))
all_noisy_objs = [s.split('.')[0] for s in all_noisy_objs]

final_rels = {}
for lvis, vg in lvis_vg_map.items():
    final_data = {}
    for vg_obj in vg:
        if (vg_obj not in all_noisy_objs):
            continue
        else:
            data = filtered_subjects[vg_obj]

            if (len(final_data) == 0):
                final_data = data
            else:
                for rel, rel_objs in data.items():
                    final_data = add_or_append(final_data, rel, rel_objs)

    final_rels[lvis] = final_data

# ============================================================================
"""
final_rels still has vg_obj2 in the value.

This code block converts vg_obj2 into lvis_obj through lvis_vg_map.
refined_final_objs save (k, v) like (lvis_obj, {'on': {lvis_obj2: number_of_times}})
"""
all_noisy_subjs = sorted(list(final_rels.keys()))

refined_final_objs = {}
for lvis_obj in all_noisy_subjs:
    final_data = {}

    data = final_rels[lvis_obj]

    for rel, rel_objs in data.items():
        final_data[rel] = {}
        subj_list = rel_objs.keys()

        for subj in subj_list:
            lvis_key = [list(lvis_vg_map.keys())[list(lvis_vg_map.values()).index(
                vg_objs)] for ind, vg_objs in enumerate(list(lvis_vg_map.values())) if subj in vg_objs][0]

            if (lvis_key not in final_data[rel].keys()):
                final_data[rel][lvis_key] = data[rel][subj]
            else:
                final_data[rel][lvis_key] += data[rel][subj]

    refined_final_objs[lvis_obj] = final_data

# ===========================================================================
"""
There are alias/synonyms in the VG relationship file.

merge the relationship in refined_final_objs with synonym relationships
and save in refined_final_rels.

e.g.

"""
print("Done object pruning. Starting relationship pruning...")
with open(f'{vg_folder}/relationship_alias.txt') as f:
    rel_list = f.readlines()

rel_list = [s.strip('\n') for s in rel_list]

rel_alias_map = {}

for rel in rel_list:
    rel_key = rel.split(',')[0]

    if (rel_key not in rel_alias_map.keys()):
        rel_alias_map[rel_key] = list(rel.split(','))
    else:
        rel_alias_map[rel_key] = list(
            set(rel_alias_map[rel_key] + list(rel.split(','))))


all_objs = list(refined_final_objs.keys())

refined_final_rels = {}
for obj_file in all_objs:
    final_data = {}

    data = refined_final_objs[obj_file]

    for rel, subj_dict in data.items():
        if (rel == ""):
            continue
        else:
            rel = rel.replace("  ", " ")

        try:
            rel_key = [list(rel_alias_map.keys())[list(rel_alias_map.values()).index(subj_rels)]
                       for ind, subj_rels in enumerate(list(rel_alias_map.values())) if rel in subj_rels][0]
        except (IndexError) as e:
            final_data[rel] = subj_dict

        if (rel_key not in final_data.keys()):
            final_data[rel_key] = subj_dict
        else:
            for subj in subj_dict.keys():
                if (subj not in final_data[rel_key].keys()):
                    final_data[rel_key][subj] = subj_dict[subj]
                else:
                    final_data[rel_key][subj] += subj_dict[subj]

    refined_final_rels[obj_file] = final_data

# ======================================================================================
all_objs = list(refined_final_rels.keys())
all_objs_list = sorted([s.split('.')[0] for s in all_objs])
all_rels_list = []  # list of all the relations
obj_rel_list = []  # list of all the objects

for obj_file in all_objs:
    data = refined_final_rels[obj_file]

    for rel, _ in data.items():
        obj_rel_list.append(obj_file.split('.')[0])
        all_rels_list.append(rel)

obj_rel_count_dict = Counter(obj_rel_list)
obj_rel_count_dict = {k: v for k, v in sorted(
    obj_rel_count_dict.items(), key=lambda item: item[1], reverse=True)}

rel_count_dict = Counter(all_rels_list)
rel_count_dict = {k: v for k, v in sorted(
    rel_count_dict.items(), key=lambda item: item[1], reverse=True)}

all_rels_list = sorted(list(set(all_rels_list)))

'''
>>> rel_count_dict
{'on': 18, 'near': 10, 'above': 9, 'has': 6, 'in': 3, 'and': 2, 'in front of': 2, 'of': 2, 'sitting on': 2, 'under': 2, 'before': 1, 'at': 1, 'in front': 1, 'made of': 1, 'empty': 1, 'a': 1, 'on top': 1, 'front': 1, 'with': 1, 'sits on': 1, 'sitting in': 1, 'on back of': 1, 'to right of': 1, 'atop': 1, 'for': 1, 'behind': 1, 'against': 1, 'sign on wall': 1, 'full of': 1}
>>> obj_rel_count_dict
{'Chair': 8, 'CD': 4, 'Pen': 4, 'Cabinet': 4, 'Fork': 3, 'Mug': 3, 'Lamp': 3, 'Vase': 3, 'Apple': 3, 'Pillow': 3, 'Knife': 3, 'Curtains': 3, 'Laptop': 3, 'Bowl': 2, 'Window': 2, 'Book': 2, 'Cup': 2, 'Cloth': 2, 'Plate': 2, 'Painting': 2, 'Drawer': 2, 'Box': 2, 'Potato': 2, 'Egg': 1, 'Sofa': 1, 'Blinds': 1, 'Shelf': 1, 'TableTop': 1, 'Spoon': 1, 'TeddyBear': 1, 'Newspaper': 1}
'''

# =================================================================================
"""
summarize the number of times two objects have relationships.

top_subject_relationships have (k, v) like:
'wineglass': {'table': 329, 'plate': 4, 'desk': 3, 'wineglass': 2, 'basket': 1, 
'bottle': 1, 'tablecloth': 1, 'coaster': 1, 'box': 1}
"""
print("Done relationship pruning. Saving top relationships...")
all_objs = list(refined_final_rels.keys())
all_objs = sorted([s.split('.')[0] for s in all_objs])

top_subject_relationships = {}
for obj in all_objs:
    obj_rel_dict = {}
    data = refined_final_rels[obj]

    for rel, subj_dict in data.items():
        if (obj_rel_dict is None):
            obj_rel_dict = subj_dict
        else:
            for subj in subj_dict.keys():
                if (subj not in obj_rel_dict.keys()):
                    obj_rel_dict[subj] = subj_dict[subj]
                else:
                    obj_rel_dict[subj] += subj_dict[subj]

    obj_rel_dict = {k: v for k, v in sorted(
        obj_rel_dict.items(), key=lambda item: item[1], reverse=True)}

    top_subject_relationships[obj] = obj_rel_dict
    #print(f'obj = {obj}, obj_rel_dict = {obj_rel_dict}')


# ======================
"""
merge lvis_obj synonyms
"""
with bz2.BZ2File(f'{output_folder}/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
    LVIS_dict = cPickle.load(fp)
    lvis_cat_list = LVIS_dict['cat_synonyms']
    lvis_rowid_to_catid_dict = LVIS_dict['rowid2catid_dict']

lvis_cat_relationships = {}
for obj1 in sorted(list(top_subject_relationships)):
    data = top_subject_relationships[obj1]
    obj1_id = lvis_rowid_to_catid_dict[lvis_cat_list.index(obj1)]

    if obj1_id in lvis_cat_relationships:
        new_data = lvis_cat_relationships[obj1_id]
    else:
        new_data = {}
    for obj2 in sorted(list(data.keys())):
        obj2_id = lvis_rowid_to_catid_dict[lvis_cat_list.index(obj2)]
        if obj2_id in new_data:
            new_data[obj2_id] += data[obj2]
        else:
            new_data[obj2_id] = data[obj2]

    lvis_cat_relationships[obj1_id] = new_data


# ======================================
"""
Build the adjacency matrix 
"""

'''
all_obj = sorted(list(top_subject_relationships.keys()))
adjacency_obj = np.zeros((len(all_obj), len(all_obj)), dtype=int)

for i, obj_file in enumerate(all_obj):
    data = top_subject_relationships[obj_file]
    subjects = list(data.keys())
    for obj in subjects:
        if (data[obj] >= 5):
            adjacency_obj[i][all_obj.index(f"{obj}")] = 1
'''


all_obj_cat_ids = sorted(list(lvis_cat_relationships.keys()))
adjacency_cat_id = np.zeros(
    (len(all_obj_cat_ids), len(all_obj_cat_ids)), dtype=np.int16)

for i, obj_cat_id in enumerate(all_obj_cat_ids):
    data = lvis_cat_relationships[obj_cat_id]
    subjects = list(data.keys())
    for obj2_cat_id in subjects:
        if (data[obj2_cat_id] >= 5):
            adjacency_cat_id[i][all_obj_cat_ids.index(obj2_cat_id)] = 1

LVIS_relationships = {}
LVIS_relationships['cat_id_relationship_dict'] = lvis_cat_relationships
LVIS_relationships['adjacency_cat_id'] = adjacency_cat_id
LVIS_relationships['all_obj_cat_ids'] = all_obj_cat_ids
with bz2.BZ2File(f'{output_folder}/LVIS_relationships.pbz2', 'w') as fp:
    cPickle.dump(
        LVIS_relationships,
        fp
    )
