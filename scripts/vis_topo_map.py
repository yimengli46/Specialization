'''
visualize the topological graph built for Navid CLIP experiment

'''

import numpy as np
import matplotlib.pyplot as plt
import random
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, pxl_coords_to_pose, gen_arrow_head_marker, read_occ_map_npy, plus_theta_fn, minus_theta_fn, convertInsSegToSSeg, read_map_npy, wrap_angle
from core import cfg
from random import Random
from modeling.utils.navigation_utils import get_obs_and_pose
from modeling.utils.map_utils_pcd_height import SemanticMap
import habitat
import os
import math
import bz2
import _pickle as cPickle
import cv2
import skimage.measure
from math import floor
from modeling.utils.navigation_utils import change_brightness
from skimage.draw import line
import sknw
from skimage.morphology import skeletonize


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


class TOPO:

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

        self.device_id = 0
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

        # ============================ build edges ===============================

        # traverse every pair of nodes, if there are no obstacle between them, add an edge
        edges = []

        for i in range(all_node_coords.shape[1]):
            for j in range(i + 1, all_node_coords.shape[1]):
                source_node = all_node_coords[:, i:i + 1]
                end_node = all_node_coords[:, j:j + 1]
                # check obstacle between source node and end node
                rr_line, cc_line = line(
                    source_node[1, 0], source_node[0, 0], end_node[1, 0], end_node[0, 0])
                line_vals = gt_occ_map[rr_line, cc_line]
                if np.all(line_vals):
                    edges.append([i, j])
                    edges.append([j, i])

        # =================== go through each node, check if edge have close angle ================
        for i in range(all_node_coords.shape[1]):
            unwanted_edges = []
            current_node_is_source_edges = []
            for edge in edges:
                a, b = edge
                if a == i:
                    current_node_is_source_edges.append(edge)

            # print(
            #     f'node = {all_node_coords[:, i]}, current_node_edges = {current_node_is_source_edges}  ')

            # check if the edge angle is close
            num_edges = len(current_node_is_source_edges)
            if num_edges > 1:
                dists = []
                for edge in current_node_is_source_edges:
                    a, b = edge
                    a = all_node_coords[:, a:a + 1]
                    b = all_node_coords[:, b:b + 1]
                    dist = (a[1, 0] - b[1, 0])**2 + (a[0, 0] - b[0, 0])**2
                    dists.append(dist)
                # print(f'dists = {dists}')

                # sort the edges
                dists = np.array(dists)
                edge_idxs = np.argsort(dists)
                # traverse from the short edge to the long edges
                for j, edge_i0 in enumerate(edge_idxs[:-1]):
                    for edge_i1 in edge_idxs[j + 1:]:
                        a1, b1 = current_node_is_source_edges[edge_i0]
                        a2, b2 = current_node_is_source_edges[edge_i1]
                        a1_node = all_node_coords[:, a1:a1 + 1]
                        b1_node = all_node_coords[:, b1:b1 + 1]
                        a2_node = all_node_coords[:, a2:a2 + 1]
                        b2_node = all_node_coords[:, b2:b2 + 1]
                        angle1 = math.atan2(
                            b1_node[1, 0] - a1_node[1, 0], b1_node[0, 0] - a1_node[0, 0])
                        angle2 = math.atan2(
                            b2_node[1, 0] - a2_node[1, 0], b2_node[0, 0] - a2_node[0, 0])
                        angle_diff = abs(wrap_angle(angle1 - angle2))
                        # print(
                        #    f'edge1 = {current_node_is_source_edges[edge_i0]}, edge2 = {current_node_is_source_edges[edge_i1]}, angle_diff = {angle_diff}')
                        if angle_diff <= np.pi/6:
                            # if [a2, b2] not in unwanted_edges:
                            # print(f'=> delete this edge')
                            unwanted_edges.append([a2, b2])
                            num_edges -= 1

            for unwanted_edge in unwanted_edges:
                if unwanted_edge in edges:
                    edges.remove(unwanted_edge)

        # === go through each node, if node is not in any edge, find the nearest neighbor and connect to it ===
        mask_known = semantic_occupancy_map > 0
        for i in range(all_node_coords.shape[1]):
            current_node_edges = []

            for edge in edges:
                a, b = edge
                if a == i or b == i:
                    current_node_edges.append(edge)

            if len(current_node_edges) == 0:
                current_node = all_node_coords[:, i:i + 1]
                dist = np.sqrt(
                    ((current_node - all_node_coords)**2).sum(axis=0))
                node_idxs = np.argsort(dist)
                # go through each node
                for node_idx in node_idxs:
                    if node_idx == i:
                        continue
                    rr_line, cc_line = line(
                        current_node[1, 0], current_node[0, 0],
                        all_node_coords[:, node_idx:node_idx + 1][1, 0],
                        all_node_coords[:, node_idx:node_idx + 1][0, 0])
                    line_vals = mask_known[rr_line, cc_line]
                    if np.all(line_vals):
                        edges.append([i, node_idx])
                        break

        # visualize the topological map
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
        ax.imshow(self.color_semantic_map)

        # draw the topological nodes
        x = all_node_coords[0, :].flatten()
        y = all_node_coords[1, :].flatten()
        edges = np.array(edges)
        ax.plot(x[edges.T], y[edges.T], linestyle='-',
                color='y', markerfacecolor='red', marker='o', zorder=1)
        ax.scatter(
            x=all_node_coords[0, :], y=all_node_coords[1, :], c='red', s=50, zorder=2)

        # draw the text
        x, y = [], []
        for ins in self.list_instances:
            center = ins['center']
            cat = ins['cat']

            x.append(center[0])
            y.append(center[1])

            try:
                cat_name = self.idx2cat_dict[cat]
            except:
                cat_name = 'unknown'
            ax.text(center[0], center[1], cat_name)

        ax.scatter(x=x, y=y, c='b', s=20)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()

        plt.show()


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

output_folder = 'output/vis_topo'
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

data = TOPO(split=split, scene_floor_tuple=scene_floor_tuple,
            saved_dir=split_folder, hm3d_to_lvis_dict=hm3d_to_lvis_dict, LVIS_dict=LVIS_dict)
