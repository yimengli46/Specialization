"""
densely sample locations on the occupancy map
collect panoramaic views at sampled locations
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from modeling.utils.baseline_utils import convertInsSegToSSeg, create_folder
import habitat
import habitat_sim
import random
from modeling.utils.baseline_utils import pose_to_coords, read_occ_map_npy
from core import cfg
import json
import bz2
import _pickle as cPickle

SEED = cfg.GENERAL.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)

split = 'val'
output_folder = f'output/densely_sample_locations_for_render_panor/{split}'
semantic_map_folder = f'output/semantic_map/{split}'

list_scene_idx = [5]

# after testing, using 8 angles is most efficient
theta_lst = [0]
built_scenes = []
cell_size = 1.

scene_floor_dict = np.load(
    f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy', allow_pickle=True).item()

# ============================= build a grid =========================================
x = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, cell_size)
z = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, cell_size)
xv, zv = np.meshgrid(x, z)
grid_H, grid_W = zv.shape

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
    config = habitat.get_config(
        config_paths=cfg.GENERAL.DATALOADER_CONFIG_PATH)
    config.defrost()
    config.SIMULATOR.SCENE = f'data/scene_datasets/hm3d/{split}/{scene_with_index}/{scene_with_index[6:]}.basis.glb'
    config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
    config.SIMULATOR.SCENE_DATASET = f'data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    config.freeze()

    env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
    env.reset()

    # ============================ get scene ins to cat dict
    scene = env.semantic_annotations()
    ins2cat_dict = {
        int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}

    category_id_to_name_dict = {obj.category.index(
    ): obj.category.name() for obj in scene.objects}

    scene_dict = scene_floor_dict[scene_with_index]

    # =============================== traverse each floor ===========================
    for floor_id in list(scene_dict.keys()):
        height = scene_dict[floor_id]['y']
        scene_name = f'{scene_with_index}_{floor_id}'

        # =============================== traverse each floor ===========================
        print(f'*****scene_name = {scene_name}***********')

        saved_folder = f'{output_folder}/{scene_name}'
        create_folder(saved_folder, clean_up=False)

        np.save(f'{saved_folder}/category_id_to_name_dict.npy',
                category_id_to_name_dict)

        '''
		npy_file = f'{saved_folder}/occ_map_sampled_locations.jpg'
		if os.path.isfile(npy_file):
			print(f'!!!!!!!!!!!!!!!!!!!!occ map sampled locations file exists. skip scene {scene_name}')
			continue
		'''

        occ_map_npy = np.load(
            f'output/semantic_map/{split}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
        gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(
            occ_map_npy)

        occ_map = np.zeros((grid_H, grid_W), dtype=int)

        count_ = 0
        sampled_locations = []
        # ========================= densely sample locations ===========================
        for grid_z in range(grid_H):
            for grid_x in range(grid_W):

                x = xv[grid_z, grid_x]
                z = zv[grid_z, grid_x]
                y = height

                agent_pos = np.array([x, y, z])
                flag_nav = env.is_navigable(agent_pos)

                if flag_nav:
                    # ================= render observation =======================
                    theta_lst = [0, pi/2, pi, pi*3./2]
                    agent_pos = np.array([x, y, z])
                    eps_data_rgb = {}
                    eps_data_sseg = {}
                    for idx_theta, theta in enumerate(theta_lst):
                        agent_rot = habitat_sim.utils.common.quat_from_angle_axis(
                            theta, habitat_sim.geo.GRAVITY)
                        obs = env.get_observations_at(
                            agent_pos, agent_rot, keep_agent_at_new_pose=False)
                        eps_data_rgb[idx_theta] = obs['rgb'].copy()
                        InsSeg_img = obs["semantic"]
                        sseg_img = convertInsSegToSSeg(
                            InsSeg_img, ins2cat_dict)
                        eps_data_sseg[idx_theta] = sseg_img.copy()
                        # plt.imshow(sseg_img)
                        # plt.show()

                    sample_name = str(count_).zfill(5)
                    with bz2.BZ2File(f'{saved_folder}/{sample_name}.pbz2', 'w') as fp:
                        cPickle.dump(eps_data_rgb, fp)
                        cPickle.dump(eps_data_sseg, fp)

                    count_ += 1

                    # ===================== save the sampled locations ===============
                    x = xv[grid_z, grid_x]
                    z = zv[grid_z, grid_x]
                    # should be map pose
                    z = -z
                    x_coord, z_coord = pose_to_coords(
                        (x, z), pose_range, coords_range, WH, flag_cropped=True)
                    sampled_locations.append((x_coord, z_coord))

        # ============================ save the sampled locations ======================
        eps_data = {}
        eps_data['locations'] = sampled_locations
        # '''
        with bz2.BZ2File(f'{saved_folder}/sampled_locations.pbz2', 'w') as fp:
            cPickle.dump(
                eps_data,
                fp
            )
        # '''
        #np.save(f'{saved_folder}/sampled_locations.npy', eps_data)

        # ====================== draw the sampled locations on the map ================================
        x_coord_lst, z_coord_lst = zip(*sampled_locations)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.imshow(gt_occ_map, cmap='gray')
        ax.scatter(x_coord_lst, z_coord_lst,
                   marker='o', s=50, c='blue', zorder=5)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #ax.set_title('improved observed_occ_map + frontiers')

        fig.tight_layout()
        plt.title('observed area')
        # plt.show()
        fig.savefig(f'{saved_folder}/occ_map_sampled_locations.jpg')
        plt.close()

    env.close()
