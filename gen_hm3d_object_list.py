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
    lvis_cat_list = LVIS_dict['cat_synonyms']
    lvis_rowid_to_catid_dict = LVIS_dict['rowid2catid_dict']
    lvis_cat_embedding = LVIS_dict['cat_synonyms_embedding']

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

# ======================= remove the unwanted hm3d categories ===================
hm3d_to_lvis_dict.pop('air duct')
hm3d_to_lvis_dict.pop('air vent')
hm3d_to_lvis_dict.pop('air vent fan')
hm3d_to_lvis_dict.pop('ball pool')
hm3d_to_lvis_dict.pop('banner')
hm3d_to_lvis_dict.pop('bar')
hm3d_to_lvis_dict['basket of towels'] = 'basket'
hm3d_to_lvis_dict.pop('bath utensil')
hm3d_to_lvis_dict.pop('bath wall')
hm3d_to_lvis_dict['bathmat'] = 'bath_mat'
hm3d_to_lvis_dict.pop('bathroom stuff')
hm3d_to_lvis_dict.pop('bedframe')
hm3d_to_lvis_dict.pop('bin')
hm3d_to_lvis_dict.pop('board')
hm3d_to_lvis_dict.pop('boards')
hm3d_to_lvis_dict.pop('book shelf')
hm3d_to_lvis_dict.pop('bookshelf')
hm3d_to_lvis_dict['bowl of sweets'] = 'bowl'
hm3d_to_lvis_dict['cabidet'] = 'cabinet'
hm3d_to_lvis_dict.pop('can of paint')
hm3d_to_lvis_dict.pop('cans of paint')
hm3d_to_lvis_dict['carpet roll'] = 'runner_(carpet)'
hm3d_to_lvis_dict.pop('ceiling fan vent')
hm3d_to_lvis_dict.pop('ceiling pipe')
hm3d_to_lvis_dict.pop('ceiling pipes')
hm3d_to_lvis_dict.pop('column')
hm3d_to_lvis_dict.pop('counter')
hm3d_to_lvis_dict.pop('cross')
hm3d_to_lvis_dict.pop('decorative cloth')
hm3d_to_lvis_dict.pop('door')
hm3d_to_lvis_dict.pop('door  hinge')
hm3d_to_lvis_dict.pop('door framr')
hm3d_to_lvis_dict.pop('door handle')
hm3d_to_lvis_dict.pop('door hinge')
hm3d_to_lvis_dict.pop('door knob')
hm3d_to_lvis_dict.pop('door mat')
hm3d_to_lvis_dict.pop('door slide')
hm3d_to_lvis_dict.pop('doors')
hm3d_to_lvis_dict['drum'] = 'drum_(musical_instrument)'
hm3d_to_lvis_dict.pop('elevator')
hm3d_to_lvis_dict.pop('file binder')
hm3d_to_lvis_dict.pop('floor vent')
hm3d_to_lvis_dict.pop('fruit')
hm3d_to_lvis_dict.pop('fruits')
hm3d_to_lvis_dict.pop('furniture')
hm3d_to_lvis_dict.pop('garage door motor')
hm3d_to_lvis_dict.pop('grate')
hm3d_to_lvis_dict.pop('heat vent')
hm3d_to_lvis_dict['jewlery box'] = 'jewelry'
hm3d_to_lvis_dict.pop('lid')
hm3d_to_lvis_dict.pop('oven vent')
hm3d_to_lvis_dict.pop('pitcher')
hm3d_to_lvis_dict.pop('rack')
hm3d_to_lvis_dict.pop('rail')
hm3d_to_lvis_dict.pop('shower door')
hm3d_to_lvis_dict.pop('shower glass')
hm3d_to_lvis_dict.pop('shower knob')
hm3d_to_lvis_dict.pop('shower pipe')
hm3d_to_lvis_dict.pop('sideboard')
hm3d_to_lvis_dict.pop('sign')
hm3d_to_lvis_dict.pop('skirting board')
hm3d_to_lvis_dict.pop('skylight')
hm3d_to_lvis_dict.pop('socket')
hm3d_to_lvis_dict['soft chair'] = 'sofa'
hm3d_to_lvis_dict.pop('stair step')
hm3d_to_lvis_dict.pop('stone')
hm3d_to_lvis_dict.pop('stones')
hm3d_to_lvis_dict['stuffed duck'] = 'toy'
hm3d_to_lvis_dict['table on wheels'] = 'table'
hm3d_to_lvis_dict.pop('tank')
hm3d_to_lvis_dict['toilet cabinet'] = 'cabinet'
hm3d_to_lvis_dict.pop('toilet cleaner')
hm3d_to_lvis_dict['toilet handle'] = 'toilet'
hm3d_to_lvis_dict['toilet paper dispenser'] = 'toilet_paper'
hm3d_to_lvis_dict.pop('toilet plunger')
hm3d_to_lvis_dict.pop('toiletry')
hm3d_to_lvis_dict.pop('toiletry bag')
hm3d_to_lvis_dict.pop('towel paper dispenser')
hm3d_to_lvis_dict['toy airplane'] = 'toy'
hm3d_to_lvis_dict.pop('tree')
hm3d_to_lvis_dict['tv table'] = 'table'
hm3d_to_lvis_dict.pop('unknown/ probably fan vent')
hm3d_to_lvis_dict.pop('vegetables')
hm3d_to_lvis_dict.pop('vent')
hm3d_to_lvis_dict.pop('ventialtion')
hm3d_to_lvis_dict.pop('ventilation')
hm3d_to_lvis_dict.pop('ventilation hood')
hm3d_to_lvis_dict.pop('ventilator')
hm3d_to_lvis_dict.pop('vessel')
hm3d_to_lvis_dict.pop('wal')
hm3d_to_lvis_dict.pop('wall')
hm3d_to_lvis_dict.pop('wall cubby')
hm3d_to_lvis_dict.pop('wall vent')
hm3d_to_lvis_dict.pop('washing stuff')
hm3d_to_lvis_dict.pop('wine rack')
hm3d_to_lvis_dict.pop('śign')


np.save(f'{output_folder}/hm3d_to_lvis_dict.npy', hm3d_to_lvis_dict)
