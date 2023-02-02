import json
import difflib

set_categories = set()


split = 'train'
output_folder = f'output/knowledge_graph'
#semantic_map_folder = f'output/semantic_map/{split}'


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

                set_categories.add(str(word))

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

                #print(f'word: {word}')

                set_categories.add(str(word).strip().lower())


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

'''
f= open(f"{output_folder}/HM3D_categories.txt","w+")
for i in range(len(obj_list)-1):
    f.write(sorted(obj_list)[i]+'\n')
f.write(sorted(obj_list)[-1])
f.close()
'''

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
