import numpy as np
import matplotlib.pyplot as plt
import random
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, pxl_coords_to_pose, gen_arrow_head_marker, read_occ_map_npy, plus_theta_fn, minus_theta_fn, convertInsSegToSSeg, read_map_npy
from core import cfg
from random import Random
from modeling.utils.navigation_utils import get_obs_and_pose
from modeling.utils.map_utils_pcd_height import SemanticMap
import habitat
import os
import math
import bz2
import _pickle as cPickle
import argparse
import multiprocessing
import json
import cv2
import skimage.measure
from math import floor
from modeling.utils.navigation_utils import change_brightness
from skimage.draw import line
import sknw
from skimage.morphology import skeletonize
import skimage.measure
import matplotlib.patches as patches


def get_img_coordinates(img):
    H, W = img.shape[:2]
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack((xv, yv), axis=2)
    return coords


def build_env(split, scene_with_index, device_id=0):
    # ================================ load habitat env============================================
    print(f'scene_with_index = {scene_with_index}')
    config = habitat.get_config(
        config_paths='configs/habitat_env/dataloader_res1024.yaml')
    config.defrost()
    config.SIMULATOR.SCENE = f'data/scene_datasets/hm3d/{split}/{scene_with_index}/{scene_with_index[6:]}.glb'
    config.SIMULATOR.SCENE_DATASET = 'data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = device_id
    config.freeze()
    env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
    return env


def prune_skeleton_graph(skeleton_G):
    dict_node_numEdges = {}
    for edge in skeleton_G.edges():
        u, v = edge
        for node in [u, v]:
            if node in dict_node_numEdges:
                dict_node_numEdges[node] += 1
            else:
                dict_node_numEdges[node] = 1
    to_prune_nodes = []
    for k, v in dict_node_numEdges.items():
        if v < 2:
            to_prune_nodes.append(k)
    skeleton_G_pruned = skeleton_G.copy()
    skeleton_G_pruned.remove_nodes_from(to_prune_nodes)
    return skeleton_G_pruned


class Data_Gen_View:

    def __init__(self, split, scene_floor_tuple, saved_dir='', hm3d_to_lvis_dict=None,
                 LVIS_dict=None):
        self.split = split
        self.scene_name, self.floor_id, self.height = scene_floor_tuple
        self.random = Random(cfg.GENERAL.RANDOM_SEED)

        # ============= create scene folder =============
        scene_folder = f'{saved_dir}/{self.scene_name}_{self.floor_id}'
        if not os.path.exists(scene_folder):
            os.mkdir(scene_folder)
        self.scene_folder = scene_folder

        self.cell_size = 1.  # location sampling granularity
        self.thresh_vicinity = 2. / cfg.SEM_MAP.CELL_SIZE

        print(f'init new scene: {self.scene_name}')

        if cfg.PRED.VIEW.multiprocessing == 'single':
            self.device_id = 0
        elif cfg.PRED.VIEW.multiprocessing == 'mp':
            self.device_id = gpu_Q.get()
        # ================================ load habitat env============================================
        self.env = build_env(
            self.split, self.scene_name, device_id=self.device_id)
        self.env.reset()

        scene = self.env.semantic_annotations()
        self.ins2cat_dict = {
            int(obj.id.split("_")[-1]): obj.category.index()
            for obj in scene.objects
        }

        # ================= processing semantics ================
        self.hm3d_to_lvis_dict = hm3d_to_lvis_dict
        self.LVIS_dict = LVIS_dict

        goal_obj_list = sorted(
            list(set(hm3d_to_lvis_dict.values())))  # size: 351
        self.goal_obj_index_list = list(set(self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
            cat_syn)] for cat_syn in goal_obj_list))  # size: 310
        self.num_cats = len(self.goal_obj_index_list)
        print(f'num_cats = {self.num_cats}')

        print(
            f'start dataset on scene {self.scene_name} floor {self.floor_id} =>')

        # ============================= build scene category list =============================
        print('build scene category list ...')
        # load the category id to name dict
        self.idx2cat_dict = np.load(
            f'output/semantic_map/{self.split}/{self.scene_name}_{self.floor_id}/category_id_to_name_dict.npy',
            allow_pickle=True).item()
        # convert hm3d category into goal categories
        self.goal_category_set = set()
        self.ignored_category_id_set = set()
        for k, v in self.idx2cat_dict.items():
            v = v.strip().lower()
            if v in list(hm3d_to_lvis_dict.keys()):
                lvis_goal_obj = hm3d_to_lvis_dict[v]
                self.idx2cat_dict[k] = lvis_goal_obj
                self.goal_category_set.add(lvis_goal_obj)
            #     print(f'v = {v}, new_v = {idx2cat_dict[k]}')
            else:
                self.idx2cat_dict[k] = v
                self.ignored_category_id_set.add(k)
                print(f'class {v} is not a goal object.')

        # =========================== load pre-built semantic and occupancy maps ========================
        print('load maps ...')
        sem_map_npy = np.load(
            f'output/semantic_map/{self.split}/{self.scene_name}_{self.floor_id}/BEV_semantic_map.npy',
            allow_pickle=True).item()
        gt_sem_map, self.pose_range, self.coords_range, self.WH = read_map_npy(
            sem_map_npy)
        occ_map_npy = np.load(
            f'output/semantic_map/{self.split}/{self.scene_name}_{self.floor_id}/BEV_occupancy_map.npy',
            allow_pickle=True).item()
        gt_occ_map, _, _, _ = read_occ_map_npy(occ_map_npy)

        # ======================== build the object centroid map =======================
        semantic_occupancy_map = gt_sem_map.copy()
        self.H, self.W = semantic_occupancy_map.shape
        semantic_occupancy_map = cv2.resize(semantic_occupancy_map, (
            self.W * cfg.SEM_MAP.VIS_ENLARGE_RATIO, self.H * cfg.SEM_MAP.VIS_ENLARGE_RATIO),
            interpolation=cv2.INTER_NEAREST)
        large_H, large_W = semantic_occupancy_map.shape
        x = np.linspace(0, large_W - 1, large_W)
        y = np.linspace(0, large_H - 1, large_H)
        xv, yv = np.meshgrid(x, y)

        # ====================================== compute centers of semantic classes =====================================
        print('compute object centers ...')
        cat_binary_map = semantic_occupancy_map.copy()
        for cat in self.ignored_category_id_set:
            cat_binary_map = np.where(
                cat_binary_map == cat, -1, cat_binary_map)
        # run skimage to find the number of objects belong to this class
        self.instance_label, num_ins = skimage.measure.label(
            cat_binary_map, background=-1, connectivity=1, return_num=True)

        self.list_instances = []
        for idx_ins in range(1, num_ins + 1):
            mask_ins = (self.instance_label == idx_ins)
            if np.sum(mask_ins) >= 1:  # should have at least 20 pixels
                x_coords = xv[mask_ins]
                y_coords = yv[mask_ins]
                ins_center = (floor(np.median(x_coords)),
                              floor(np.median(y_coords)))
                ins_cat = semantic_occupancy_map[int(
                    y_coords[0]), int(x_coords[0])]
                if self.idx2cat_dict[ins_cat] in self.goal_category_set:
                    ins = {}
                    ins['center'] = (ins_center[0] // cfg.SEM_MAP.VIS_ENLARGE_RATIO,
                                     ins_center[1] // cfg.SEM_MAP.VIS_ENLARGE_RATIO)
                    ins['cat'] = ins_cat
                    ins['id_ins'] = idx_ins
                    ins['mask'] = mask_ins
                    self.list_instances.append(ins)

        # ================== colorize the semantic map and merge with occupancy map ==================
        self.semantic_occupancy_map = gt_sem_map.copy()
        self.color_semantic_map = apply_color_to_map(
            self.semantic_occupancy_map, type_categories='LVIS')
        # turn free space into white
        self.color_semantic_map[gt_occ_map > 0] = np.ones(3) * 255

        # ==================== build skeleton =======================
        skeleton = skeletonize(gt_occ_map)
        graph = sknw.build_sknw(skeleton)
        graph = prune_skeleton_graph(graph)
        edges_nodes = np.zeros((0, 2), dtype=np.int16)
        for (s, e) in graph.edges():
            ps = graph[s][e]['pts']
            ps_sparse = ps[20:ps.shape[0]:20, :]
            edges_nodes = np.concatenate((edges_nodes, ps_sparse))
        edges_nodes = edges_nodes.transpose()

        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes]).transpose()

        node_coords = np.concatenate((ps, edges_nodes), axis=1)[
            [1, 0], :].astype(np.int32)

        # ======================== remove red nodes that are close to themselves =============
        mask = np.ones(node_coords.shape[1], dtype=bool)
        for i in range(node_coords.shape[1]):
            current_node_coords = node_coords[:, i:i + 1]
            # print(f'current_node_coords.shape = {current_node_coords.shape}')
            dist = np.sqrt(
                ((current_node_coords - node_coords)**2).sum(axis=0))
            # print(f'dist = {dist}')
            current_mask = dist > 20
            mask[i + 1:] = current_mask[i + 1:] & mask[i + 1:]

        all_node_coords = node_coords[:, mask]

        # remove duplicates
        nodes_set = set()
        all_node_coords = all_node_coords.tolist()
        all_node_coords = list(zip(all_node_coords[0], all_node_coords[1]))
        for i in range(len(all_node_coords)):
            node = all_node_coords[i]
            nodes_set.add((node[0], node[1]))

        all_node_coords = np.zeros((2, len(nodes_set)), dtype=np.int32)
        for i, node in enumerate(nodes_set):
            all_node_coords[:, i] = node

        self.nodes = all_node_coords
        # print(f'nodes.shape = {self.nodes.shape}')

        # draw the sample locations
        if False:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            ax.imshow(self.color_semantic_map)
            ax.scatter(self.nodes[0, :], self.nodes[1, :],
                       marker='o', s=50, c='blue', zorder=5)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.tight_layout()
            plt.title('observed area')
            plt.show()

    def write_to_file(self):
        count_view = 0

        for node_i in range(self.nodes.shape[1]):

            x, z = pxl_coords_to_pose(
                self.nodes[:, node_i], self.pose_range, self.coords_range, self.WH)
            z = -z
            y = self.height

            agent_pos = np.array([x, y, z])
            # check if the current location is navigable
            if not self.env.is_navigable(agent_pos):
                # print('current pose is not navigable ...')
                continue

            for rot in [90, 180, 270, 0]:
                heading_angle = rot / 180 * np.pi
                heading_angle = plus_theta_fn(heading_angle, 0)
                obs, pose = get_obs_and_pose(
                    self.env, agent_pos, heading_angle)
                agent_map_pose = (pose[0], -pose[1], -pose[2])

                rgb_img = obs['rgb']
                depth_img = obs['depth'][:, :, 0]
                # print(f'depth_img.shape = {depth_img.shape}')
                InsSeg_img = obs["semantic"]
                if len(InsSeg_img.shape) > 2:
                    InsSeg_img = np.squeeze(InsSeg_img)
                sseg_img = convertInsSegToSSeg(
                    InsSeg_img, self.ins2cat_dict)

                semMap_module = SemanticMap(self.split, f'{self.scene_name}_{self.floor_id}',
                                            self.pose_range, self.coords_range,
                                            self.WH, self.ins2cat_dict)
                semMap_module.build_semantic_map([obs], [pose])

                semantic_map, observed_area_flag, _ = semMap_module.get_semantic_map()

                # find detected objects
                list_detected_instances = []
                for ins in self.list_instances:
                    center = ins['center']
                    cat = ins['cat']
                    ins_mask = ins['mask']

                    if np.logical_and(observed_area_flag, ins_mask).any():
                        list_detected_instances.append(ins)

                # find instances in the vicinity of detected objects
                list_vicinity_instances = []
                for ins in self.list_instances:
                    if ins not in list_detected_instances:
                        center = ins['center']

                        for detected_ins in list_detected_instances:
                            detected_center = detected_ins['center']
                            rr_line, cc_line = line(detected_center[1], detected_center[0],
                                                    center[1], center[0])
                            line_vals = self.semantic_occupancy_map[rr_line,
                                                                    cc_line] > 0
                            if line_vals.shape[0] < self.thresh_vicinity and np.all(line_vals):
                                list_vicinity_instances.append(ins)

                # ==================== build view object vector from the map ======================
                map_dist_view_to_cat = np.zeros((1, self.num_cats), dtype=np.int32)

                # first write objects in the vicinity
                # in case some objects are both detected and in the vicinity
                for ins in list_vicinity_instances:
                    cat = self.idx2cat_dict[ins['cat']]
                    cat_index = self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
                        cat)]
                    map_dist_view_to_cat[0, self.goal_obj_index_list.index(
                        cat_index)] = 2

                # detected instance has class 1
                for ins in list_detected_instances:
                    cat = self.idx2cat_dict[ins['cat']]
                    cat_index = self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
                        cat)]
                    map_dist_view_to_cat[0, self.goal_obj_index_list.index(
                        cat_index)] = 1

                # ======================== build view object vector from the detector =============
                detector_dist_view_to_cat = np.zeros((1, self.num_cats), dtype=np.int32)

                img_hm3d_cat_index_list = np.unique(sseg_img)
                bbox_list = []
                # create image coordinates
                coords = get_img_coordinates(sseg_img)
                # convert sseg_img into bbox
                for hm3d_cat_index in img_hm3d_cat_index_list:
                    lvis_cat_name = self.idx2cat_dict[hm3d_cat_index]
                    # if current category is in the goal category list
                    if lvis_cat_name in self.goal_category_set:
                        # create binary image for current category index
                        cat_binary_map = np.zeros(sseg_img.shape, dtype=np.int16)
                        cat_binary_map[sseg_img == hm3d_cat_index] = 1
                        instance_label, num_ins = skimage.measure.label(
                            cat_binary_map, background=0, connectivity=1, return_num=True)
                        # create a bbox for each mask
                        for idx_ins in range(1, num_ins + 1):
                            mask_ins = (instance_label == idx_ins)
                            mask_coords = coords[mask_ins]
                            x1 = np.min(mask_coords[:, 0])
                            x2 = np.max(mask_coords[:, 0])
                            y1 = np.min(mask_coords[:, 1])
                            y2 = np.max(mask_coords[:, 1])
                            cat_id = self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
                                lvis_cat_name)]
                            detector_dist_view_to_cat[0, self.goal_obj_index_list.index(cat_id)] = 1
                            if x2 - x1 > 5 and y2 - y1 > 5:
                                bbox_list.append([x1, y1, x2, y2, lvis_cat_name])

                # ======================== save the data ======================
                view_data = {}
                view_data['map_dist_to_cat'] = map_dist_view_to_cat
                view_data['gt_detector_dist_to_cat'] = detector_dist_view_to_cat
                view_data['rgb'] = rgb_img
                view_data['depth'] = depth_img
                view_data['sseg'] = sseg_img
                view_data['observed_area_flag'] = observed_area_flag
                view_data['agent_map_pose'] = agent_map_pose
                view_data['bbox'] = bbox_list

                sample_name = str(count_view).zfill(5)
                with bz2.BZ2File(f'{self.scene_folder}/{sample_name}.pbz2', 'w') as fp:
                    cPickle.dump(
                        view_data,
                        fp
                    )
                    print(f'count_view = {count_view}')
                    count_view += 1

                # =========================== visualize all the frontier obs and panor ====================
                if cfg.PRED.VIEW.FLAG_VIS_FRONTIER_ON_MAP:
                    x_coord, z_coord = pose_to_coords((agent_map_pose[0], agent_map_pose[1]), self.pose_range,
                                                      self.coords_range, self.WH)

                    fig = plt.figure(figsize=(20, 10))
                    gs = fig.add_gridspec(1, 3)

                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.imshow(rgb_img)
                    # draw the object detector bboxes
                    for bbox in bbox_list:
                        x1, y1, x2, y2, cat_name = bbox
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                        ax1.add_patch(rect)
                        ax1.text(x1, y1, cat_name, color='green')
                    ax1.get_xaxis().set_visible(False)
                    ax1.get_yaxis().set_visible(False)
                    ax1.set_title('egocentric view')

                    ax2 = fig.add_subplot(gs[0, 1:])
                    # draw the semantic map
                    vis_color_semantic_map = change_brightness(
                        self.color_semantic_map.copy(), observed_area_flag, value=120)
                    ax2.imshow(vis_color_semantic_map)
                    # draw the marker
                    marker, scale = gen_arrow_head_marker(
                        agent_map_pose[2])
                    ax2.scatter(x_coord,
                                z_coord,
                                marker=marker,
                                s=(30 * scale)**2,
                                c='red',
                                zorder=5)
                    # draw the object instances
                    x, y = [], []
                    for ins in list_detected_instances:
                        center = ins['center']
                        cat = ins['cat']

                        x.append(center[0])
                        y.append(center[1])

                        try:
                            cat_name = self.idx2cat_dict[cat]
                        except:
                            cat_name = 'unknown'
                        ax2.text(
                            center[0], center[1], cat_name, color='yellow')

                    ax2.scatter(x=x, y=y, c='yellow', s=30)
                    # draw the object near the detected objects
                    x, y = [], []
                    for ins in list_vicinity_instances:
                        center = ins['center']
                        cat = ins['cat']

                        x.append(center[0])
                        y.append(center[1])

                        try:
                            cat_name = self.idx2cat_dict[cat]
                        except:
                            cat_name = 'unknown'
                        ax2.text(
                            center[0], center[1], cat_name, color='green')

                    ax2.scatter(x=x, y=y, c='green', s=30)
                    ax2.get_xaxis().set_visible(False)
                    ax2.get_yaxis().set_visible(False)
                    ax2.set_title('observed area and detected instance')

                    fig.tight_layout()
                    if False:
                        plt.show()
                    else:
                        fig = plt.gcf()
                        fig.set_size_inches((11, 8.5), forward=False)
                        fig.savefig(f'{self.scene_folder}/{sample_name}.jpg',
                                    bbox_inches='tight')
                        plt.close()

        # ============== release the resources =================
        self.env.close()
        gpu_Q.put(self.device_id)
        return


def multi_run_wrapper(args):
    """ wrapper for multiprocessor """
    gen = Data_Gen_View(args[0], args[1], saved_dir=args[2],
                        hm3d_to_lvis_dict=args[3], LVIS_dict=args[4])
    gen.write_to_file()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--j', type=int, required=False, default=1)
    parser.add_argument('--split', type=str, required=True, default='val')
    args = parser.parse_args()
    cfg.merge_from_file(
        'configs/exp_gen_view_by_densely_sample_locations.yaml')
    cfg.freeze()

    # =============================== basic setup =======================================
    SEED = cfg.GENERAL.RANDOM_SEED
    random.seed(SEED)
    np.random.seed(SEED)

    split = args.split
    scene_floor_dict = np.load(
        f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy',
        allow_pickle=True).item()

    output_folder = cfg.PRED.VIEW.GEN_SAMPLES_SAVED_FOLDER
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    split_folder = f'{output_folder}/{split}'
    if not os.path.exists(split_folder):
        os.mkdir(split_folder)

    # =================================analyze json file to get the semantic files =============================
    list_scene_floor_tuple = []
    with open('data/versioned_data/hm3d-1.0/hm3d/hm3d_annotated_basis.scene_dataset_config.json') as f:
        data = json.loads(f.read())
        if split == 'val':
            list_json_dirs = data['scene_instances']['paths']['.json'][103:]
        elif split == 'train':
            list_json_dirs = data['scene_instances']['paths']['.json'][23:103]

        for json_dir in list_json_dirs:
            first_slash = json_dir.find('/')
            second_slash = json_dir.find('/', first_slash + 1)

            sem_filename = json_dir[first_slash + 1:second_slash]

            # if sem_filename == '00009-vLpv2VX547B':
            scene_dict = scene_floor_dict[sem_filename]
            for floor_id in list(scene_dict.keys()):
                height = scene_dict[floor_id]['y']
                list_scene_floor_tuple.append(
                    (sem_filename, floor_id, height))

    hm3d_to_lvis_dict = np.load(
        'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

    # load the LVIS categories
    with bz2.BZ2File(f'output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
        LVIS_dict = cPickle.load(fp)

    if cfg.PRED.VIEW.multiprocessing == 'single':  # single process
        for scene_floor_tuple in list_scene_floor_tuple:
            gen = Data_Gen_View(split, scene_floor_tuple,
                                saved_dir=split_folder, hm3d_to_lvis_dict=hm3d_to_lvis_dict, LVIS_dict=LVIS_dict)
            gen.write_to_file()
    elif cfg.PRED.VIEW.multiprocessing == 'mp':
        # ====================== get the available GPU devices ============================
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        devices = [int(dev) for dev in visible_devices]

        for device_id in devices:
            for _ in range(args.j):
                gpu_Q.put(device_id)

        with multiprocessing.Pool(processes=cfg.PRED.VIEW.NUM_PROCESS) as pool:
            args0 = [split for _ in range(len(list_scene_floor_tuple))]
            args1 = [
                scene_floor_tuple for scene_floor_tuple in list_scene_floor_tuple]
            args2 = [split_folder for _ in range(len(list_scene_floor_tuple))]
            args3 = [hm3d_to_lvis_dict for _ in range(
                len(list_scene_floor_tuple))]
            args4 = [LVIS_dict for _ in range(len(list_scene_floor_tuple))]
            pool.map(multi_run_wrapper, list(
                zip(args0, args1, args2, args3, args4)))
            pool.close()


if __name__ == "__main__":
    gpu_Q = multiprocessing.Queue()
    main()


'''
cfg.merge_from_file('configs/exp_gen_view_by_densely_sample_locations.yaml')
cfg.freeze()

SEED = cfg.GENERAL.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)

scene_name = '00800-TEEsavR23oF'
split = 'val'

scene_floor_dict = np.load(
    f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy',
    allow_pickle=True).item()

output_folder = cfg.PRED.VIEW.GEN_SAMPLES_SAVED_FOLDER
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

split_folder = f'{output_folder}/{split}'
if not os.path.exists(split_folder):
    os.mkdir(split_folder)

height = scene_floor_dict[scene_name][0]['y']
scene_floor_tuple = (scene_name, 0, height)

hm3d_to_lvis_dict = np.load(
    'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

# load the LVIS categories
with bz2.BZ2File(f'output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
    LVIS_dict = cPickle.load(fp)

data = Data_Gen_View(split=split, scene_floor_tuple=scene_floor_tuple,
                     saved_dir=split_folder, hm3d_to_lvis_dict=hm3d_to_lvis_dict, LVIS_dict=LVIS_dict)
data.write_to_file()
'''
