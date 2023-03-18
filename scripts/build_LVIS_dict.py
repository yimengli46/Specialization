"""
get LVIS embedding and categories
"""
from lvis import LVIS
import json
import numpy as np
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
cat_id_to_row_id_dict = {}
count_word = 0
for idx in range(len(data)):
    synonyms = data[idx]['synonyms']
    cat_id = data[idx]['id']
    for syn in synonyms:
        cat_list.append(syn)
        row_id_to_cat_id_dict[count_word] = cat_id
        if cat_id in cat_id_to_row_id_dict:
            cat_id_to_row_id_dict[cat_id].append(count_word)
        else:
            cat_id_to_row_id_dict[cat_id] = [count_word]
        count_word += 1

# encode with SentenseTransformer
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
cat_embedding = model.encode(cat_list)

# compute cat_id embedding by take the mean of the synonyms' embeddings belonging to the same category
cat_id_list = list(cat_id_to_row_id_dict.keys())
cat_id_embedding = np.zeros((len(cat_id_list), cat_embedding.shape[1]))
for cat_id in cat_id_list:
    row_ids = cat_id_to_row_id_dict[cat_id]
    #print(f'row_ids = {row_ids}')
    row_embeddings = cat_embedding[row_ids]
    cat_id_embedding[cat_id - 1] = np.mean(row_embeddings, axis=0)

LVIS_dict = {}
# dim: num_words (different words can belong to the same category)
LVIS_dict['cat_synonyms'] = cat_list
# dim: num_categories (small than num_words)
LVIS_dict['rowid2catid_dict'] = row_id_to_cat_id_dict
# dim: num_words (each word has an embedding)
LVIS_dict['cat_synonyms_embedding'] = cat_embedding
# dim: num_categories.
# To access embedding for one cat_id, use index (cat_id - 1)
LVIS_dict['cat_id_embedding'] = cat_id_embedding

with bz2.BZ2File(f'{output_folder}/LVIS_categories_and_embedding.pbz2', 'w') as fp:
    cPickle.dump(
        LVIS_dict,
        fp
    )
