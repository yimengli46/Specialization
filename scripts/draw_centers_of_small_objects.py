"""
draw the small object names on the semantic map
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import floor
from modeling.utils.baseline_utils import apply_color_to_map, create_folder
import random
from modeling.utils.baseline_utils import read_map_npy, read_occ_map_npy
from core import cfg
import json
import skimage.measure

SEED = cfg.GENERAL.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)

split = 'val'
output_folder = f'output/owl_detections_on_densely_sampled_locations_small_objects_only_filtered/{split}'
semantic_map_folder = f'output/semantic_map/{split}'
densely_sample_folder = f'output/densely_sample_locations_for_render_panor/{split}'

list_scene_idx = [0, 1, 2, 3, 4, 5]

scene_floor_dict = np.load(
    f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy', allow_pickle=True).item()

# =================================analyze json file to get the semantic files =============================
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

for scene_idx in range(len(list_scene_idx)):
    scene_with_index = sem_filenames[list_scene_idx[scene_idx]]
    print(f'scene = {scene_with_index}')

    scene_dict = scene_floor_dict[scene_with_index]

    # =============================== traverse each floor ===========================
    for floor_id in list(scene_dict.keys()):
        height = scene_dict[floor_id]['y']
        scene_name = f'{scene_with_index}_{floor_id}'

        # =============================== traverse each floor ===========================
        print(f'*****scene_name = {scene_name}***********')

        saved_folder = f'{output_folder}/{scene_name}'
        create_folder(saved_folder, clean_up=False)

        # =========================== load the category id to name dict ==============================
        idx2cat_dict = np.load(
            f'{densely_sample_folder}/{scene_name}/category_id_to_name_dict.npy', allow_pickle=True).item()
        # compute the ignored categories
        IGNORED_CLASS = [0]
        for k, v in idx2cat_dict.items():
            if 'wall' in v:
                IGNORED_CLASS.append(k)
            elif 'floor' in v:
                IGNORED_CLASS.append(k)
            elif 'ceiling' in v:
                IGNORED_CLASS.append(k)
            elif 'door' in v:
                IGNORED_CLASS.append(k)
            elif 'table' in v:
                IGNORED_CLASS.append(k)
            elif 'tv' in v:
                IGNORED_CLASS.append(k)
            elif 'bed' in v:
                IGNORED_CLASS.append(k)
            elif 'couch' in v:
                IGNORED_CLASS.append(k)
            elif 'window' in v:
                IGNORED_CLASS.append(k)
            elif 'stair' in v:
                IGNORED_CLASS.append(k)
            elif 'chair' in v:
                IGNORED_CLASS.append(k)
            elif 'cabinet' in v:
                IGNORED_CLASS.append(k)
            elif 'wardrobe' in v:
                IGNORED_CLASS.append(k)
            elif 'mat' in v:
                IGNORED_CLASS.append(k)
            elif 'blanket' in v:
                IGNORED_CLASS.append(k)
            elif 'clothes' in v:
                IGNORED_CLASS.append(k)
            elif 'shoes' in v:
                IGNORED_CLASS.append(k)
            elif 'shelf' in v:
                IGNORED_CLASS.append(k)
            elif 'hanger' in v:
                IGNORED_CLASS.append(k)
            elif 'pillow' in v:
                IGNORED_CLASS.append(k)
            elif 'bathtub' in v:
                IGNORED_CLASS.append(k)
            elif 'curtain' in v:
                IGNORED_CLASS.append(k)
            elif 'toilet' in v:
                IGNORED_CLASS.append(k)
            elif 'sofa' in v:
                IGNORED_CLASS.append(k)
            elif 'refrigerator' in v:
                IGNORED_CLASS.append(k)
            elif 'oven' in v:
                IGNORED_CLASS.append(k)
            elif 'frame' in v:
                IGNORED_CLASS.append(k)
            elif 'FRAME' in v:
                IGNORED_CLASS.append(k)
            elif 'sink' in v:
                IGNORED_CLASS.append(k)
            elif 'shelves' in v:
                IGNORED_CLASS.append(k)
            elif 'drawer' in v:
                IGNORED_CLASS.append(k)
            elif 'painting' in v:
                IGNORED_CLASS.append(k)
            elif 'appliance' in v:
                IGNORED_CLASS.append(k)
            elif 'dresser' in v:
                IGNORED_CLASS.append(k)
            elif 'mirror' in v:
                IGNORED_CLASS.append(k)
            elif 'worktop' in v:
                IGNORED_CLASS.append(k)
            elif 'microwave' in v:
                IGNORED_CLASS.append(k)
            elif 'book' in v:
                IGNORED_CLASS.append(k)
            else:
                print(f'v = {v}')

        # ======================================== load the semantic map =======================================
        map_npy = np.load(
            f'{semantic_map_folder}/{scene_name}/BEV_semantic_map.npy', allow_pickle=True).item()
        semantic_occupancy_map, pose_range, coords_range, WH = read_map_npy(
            map_npy)
        occ_map_npy = np.load(
            f'{semantic_map_folder}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
        gt_occ_map, _, _, _ = read_occ_map_npy(occ_map_npy)

        H, W = semantic_occupancy_map.shape
        semantic_occupancy_map = cv2.resize(semantic_occupancy_map, (int(
            W*10), int(H*10)), interpolation=cv2.INTER_NEAREST)
        H, W = semantic_occupancy_map.shape
        x = np.linspace(0, W-1, W)
        y = np.linspace(0, H-1, H)
        xv, yv = np.meshgrid(x, y)

        # ================== colorize the semantic map and merge with occupancy map ==================
        color_semantic_map = apply_color_to_map(
            semantic_occupancy_map, type_categories='LVIS')
        enlarged_occ_map = cv2.resize(
            gt_occ_map, (W, H), interpolation=cv2.INTER_NEAREST)
        # turn free space into white
        color_semantic_map[enlarged_occ_map > 0] = np.ones(3)*255

        # ====================================== compute centers of semantic classes =====================================
        # IGNORED_CLASS = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 13, 17, 18, 19, 20, 21, 27, 28, 33, 35, 36, 37, 39, 40, 42, 47, 48, 55, 56, 60, 69, 70, 71, 74, 79, 89]
        cat_binary_map = semantic_occupancy_map.copy()
        for cat in IGNORED_CLASS:
            cat_binary_map = np.where(
                cat_binary_map == cat, -1, cat_binary_map)
        # run skimage to find the number of objects belong to this class
        instance_label, num_ins = skimage.measure.label(
            cat_binary_map, background=-1, connectivity=1, return_num=True)

        list_instances = []
        for idx_ins in range(1, num_ins+1):
            mask_ins = (instance_label == idx_ins)
            if np.sum(mask_ins) > 50:  # should have at least 50 pixels
                print(f'idx_ins = {idx_ins}')
                x_coords = xv[mask_ins]
                y_coords = yv[mask_ins]
                ins_center = (floor(np.median(x_coords)),
                              floor(np.median(y_coords)))
                ins_cat = semantic_occupancy_map[int(
                    y_coords[0]), int(x_coords[0])]
                ins = {}
                ins['center'] = ins_center
                ins['cat'] = ins_cat
                list_instances.append(ins)

        # ================================== draw the instance centers and names ======================================
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
        ax.imshow(color_semantic_map)
        fig.tight_layout()
        x, y = [], []
        for ins in list_instances:
            center = ins['center']
            cat = ins['cat']

            x.append(center[0])
            y.append(center[1])

            try:
                cat_name = idx2cat_dict[cat]
            except:
                cat_name = 'unknown'
            ax.text(center[0], center[1], cat_name)

        ax.scatter(x=x, y=y, c='b', s=5)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # change figure size to the entire screen
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        fig = plt.gcf()
        fig.set_size_inches((11, 8.5), forward=False)
        # plt.show()
        fig.savefig(f'{saved_folder}/semantic_map_with_category_names.png',
                    bbox_inches='tight', dpi=100)
        plt.close()
