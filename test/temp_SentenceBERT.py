from sentence_transformers import SentenceTransformer, util
import json


model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode('unknown/ probably shower cabine')
passage_embedding = model.encode(['Coffee Maker',
                                  'coffee machine',
                                  'kitchen table', 'chair', 'kitchen table, table top', 'unknown', 'cabine'])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))
