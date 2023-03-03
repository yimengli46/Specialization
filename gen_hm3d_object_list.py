'''
find the connection between HM3D and LVIS categories
'''

import json
import difflib
import bz2
import _pickle as cPickle
from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv


set_categories = set()


split = 'train'
output_folder = f'output/knowledge_graph'

# ================================= analyze json file to get the semantic files =============================
point_filenames = []
sem_filenames = []
with open(f'data/versioned_data/hm3d-1.0/hm3d/hm3d_annotated_basis.scene_dataset_config.json') as f:
    data = json.loads(f.read())
    if split == 'val':
        list_json_dirs = data['scene_instances']['paths']['.json'][103:]
    elif split == 'train':
        list_json_dirs = data['scene_instances']['paths']['.json'][23:103]

    for json_dir in list_json_dirs:
        first_slash = json_dir.find('/')
        second_slash = json_dir.find('/', first_slash+1)

        sem_filename = json_dir[first_slash+1:second_slash]
        point_filename = json_dir[first_slash+7:second_slash]

        point_filenames.append(point_filename)
        sem_filenames.append(sem_filename)

list_scene_idx = list(range(len(sem_filenames)))

# =================================
for scene_idx in range(len(list_scene_idx)):
    scene_with_index = sem_filenames[list_scene_idx[scene_idx]]
    print(f'scene = {scene_with_index}')

    semantic_file = f'data/scene_datasets/hm3d/{split}/{scene_with_index}/{scene_with_index[6:]}.semantic.txt'

    with open(f'{semantic_file}', "r") as reader:
        for idx, line in enumerate(reader.readlines()):
            if idx > 0:
                i_first_quotes = line.find('"')
                i_second_quotes = line.find('"', i_first_quotes+1)
                word = line[i_first_quotes+1:i_second_quotes]

                #print(f'word: {word}')

                set_categories.add(str(word).strip().lower())

split = 'val'
point_filenames = []
sem_filenames = []
with open(f'data/versioned_data/hm3d-1.0/hm3d/hm3d_annotated_basis.scene_dataset_config.json') as f:
    data = json.loads(f.read())
    if split == 'val':
        list_json_dirs = data['scene_instances']['paths']['.json'][103:]
    elif split == 'train':
        list_json_dirs = data['scene_instances']['paths']['.json'][23:103]

    for json_dir in list_json_dirs:
        first_slash = json_dir.find('/')
        second_slash = json_dir.find('/', first_slash+1)

        sem_filename = json_dir[first_slash+1:second_slash]
        point_filename = json_dir[first_slash+7:second_slash]

        point_filenames.append(point_filename)
        sem_filenames.append(sem_filename)

list_scene_idx = list(range(len(sem_filenames)))

# =================================

for scene_idx in range(len(list_scene_idx)):
    scene_with_index = sem_filenames[list_scene_idx[scene_idx]]
    print(f'scene = {scene_with_index}')

    semantic_file = f'data/scene_datasets/hm3d/{split}/{scene_with_index}/{scene_with_index[6:]}.semantic.txt'

    with open(f'{semantic_file}', "r") as reader:
        for idx, line in enumerate(reader.readlines()):
            if idx > 0:
                i_first_quotes = line.find('"')
                i_second_quotes = line.find('"', i_first_quotes+1)
                word = line[i_first_quotes+1:i_second_quotes]
                set_categories.add(str(word).strip().lower())


# =================================== categorize hm3d category to LVIS category ===========================
set_categories = sorted(list(set_categories))

# load the LVIS embedding
# for each elem in the set_categories, find its cat_id
with bz2.BZ2File(f'{output_folder}/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
    LVIS_dict = cPickle.load(fp)
    lvis_cat_list = LVIS_dict['categories']
    lvis_rowid_to_catid_dict = LVIS_dict['rowid2catid_dict']
    lvis_cat_embedding = LVIS_dict['embedding']

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

thresh = 0.65
hm3d_to_lvis_dict = {}
with open(f'{output_folder}/mapping_HM3D_to_LVIS.csv', mode='w') as csv_file:
    fieldnames = ['HM3D', 'LVIS']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    for hm3d_cat in set_categories:
        query_embedding = model.encode(hm3d_cat)
        similarity = util.dot_score(
            query_embedding, lvis_cat_embedding).squeeze().numpy()
        idx_lvis_cat = np.argmax(similarity)
        max_similarity = similarity[idx_lvis_cat]
        if max_similarity >= thresh:
            print(
                f'{hm3d_cat} -> {lvis_cat_list[idx_lvis_cat]}, similarity = {max_similarity:.2f}')

            row_dict = {}
            row_dict['HM3D'] = hm3d_cat
            row_dict['LVIS'] = lvis_cat_list[idx_lvis_cat]
            writer.writerow(row_dict)

            hm3d_to_lvis_dict[hm3d_cat] = lvis_cat_list[idx_lvis_cat]

np.save(f'{output_folder}/hm3d_to_lvis_dict.npy', hm3d_to_lvis_dict)


'''
# ======================== clean up the obj list
obj_list = list(map(lambda s: s.strip().lower(), set_categories))

print(f'len(obj_list) = {len(obj_list)}')

# remove unwanted words
temp_obj_list = set()
for obj in obj_list:
    obj = obj.replace('/', ' ')
    obj = obj.replace('-', ' ')
    obj = obj.replace(' w ', ' ')
    if 'unknown' not in obj and len(obj) > 0:
        temp_obj_list.add(obj.strip())
obj_list = temp_obj_list

obj_list = sorted(obj_list)
# ============ remove synonyms
obj1_obj2_map = {}

for i, obj in enumerate(obj_list):
    a = obj_list[i]
    b = difflib.get_close_matches(
        obj, possibilities=obj_list[i+1:], n=5, cutoff=0.9)
    print(f'a = {a}, b = {b}')
    obj1_obj2_map[a] = b

for k, v in obj1_obj2_map.items():
    for word in v:
        try:
            obj_list.remove(word)
        except:
            print(f'word {word} is already removed.')


# f= open(f"{output_folder}/HM3D_categories.txt","w+")
# for i in range(len(obj_list)-1):
#     f.write(sorted(obj_list)[i]+'\n')
# f.write(sorted(obj_list)[-1])
# f.close()


# ===========================================================================================
with open('/home/yimeng/work/topo_map_specialization/MJOLNIR-master/kg_prep/kg_data/thor_v1_objects.txt') as f:
    thor_obj_list = f.readlines()

thor_obj_list = list(map(lambda s: s.strip().lower(), thor_obj_list))

hm3dobj_thorobj_map = {}

for i, obj in enumerate(thor_obj_list):
    a = thor_obj_list[i]
    b = difflib.get_close_matches(obj, possibilities=obj_list, n=5, cutoff=0.9)
    if len(b) > 0:
        print(f'a = {a}, b = {b}')
        hm3dobj_thorobj_map[a] = b
'''
