from lvis import LVIS
import json

from sentence_transformers import SentenceTransformer, util

import bz2
import _pickle as cPickle

output_folder = 'output/knowledge_graph'


'''
lvis = LVIS(annotation_path='../lvis-api/data/lvis_v1_train.json')
cat_ids = lvis.get_cat_ids()
cat_dict = lvis.load_cats(cat_ids)
json.dump(cat_dict, open(f'{output_folder}/lvis_categories.json', 'w'))
'''


# JSON file
f = open(f'{output_folder}/lvis_categories.json', "r")

# Reading from file
data = json.loads(f.read())

cat_list = []
row_id_to_cat_id_dict = {}
count_word = 0
for idx in range(len(data)):
    synonyms = data[idx]['synonyms']
    cat_id = data[idx]['id']
    for syn in synonyms:
        cat_list.append(syn)
        row_id_to_cat_id_dict[count_word] = cat_id
        count_word += 1


# encode with SentenseTransformer
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
cat_embedding = model.encode(cat_list)

LVIS_dict = {}
LVIS_dict['categories'] = cat_list
LVIS_dict['rowid2catid_dict'] = row_id_to_cat_id_dict
LVIS_dict['embedding'] = cat_embedding

with bz2.BZ2File(f'{output_folder}/LVIS_categories_and_embedding.pbz2', 'w') as fp:
    cPickle.dump(
        LVIS_dict,
        fp
    )
