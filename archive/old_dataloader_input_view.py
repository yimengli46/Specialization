import numpy as np
import matplotlib.pyplot as plt
from modeling.utils.baseline_utils import read_map_npy, apply_color_to_map, read_occ_map_npy, get_img_coordinates
import bz2
import _pickle as cPickle
from core import cfg
import scipy.ndimage
import cv2
from math import floor, ceil
import skimage.measure
from skimage.graph import MCP_Geometric as MCPG
from random import Random
import torch.utils.data as data
import os
import glob
from collections import OrderedDict
import torch


class view_dataset(data.Dataset):

    def __init__(self, split, scene_name, floor_id=0, data_folder='', hm3d_to_lvis_dict=None,
                 LVIS_dict=None):
        self.random = Random(cfg.GENERAL.RANDOM_SEED)

        self.split = split
        self.scene_name = scene_name
        self.floor_id = floor_id

        self.data_folder = data_folder

        self.hm3d_to_lvis_dict = hm3d_to_lvis_dict
        self.LVIS_dict = LVIS_dict

        print(
            f'start dataset on scene {self.scene_name} floor {self.floor_id} =>')

        # ============================= build the dict of views ===============================
        print('build the scene view dictionary.')
        sample_name_list = [os.path.splitext(os.path.basename(x))[0]
                            for x in sorted(glob.glob(f'{data_folder}/{self.scene_name}_{self.floor_id}/*.pbz2'))]
        print(f'find {len(sample_name_list)} files.')
        self.dict_sample_id_to_num_frons = OrderedDict()
        for sample_name in sample_name_list:
            with bz2.BZ2File(f'{self.data_folder}/{self.scene_name}_{self.floor_id}/{sample_name}.pbz2', 'rb') as fp:
                pk_file = cPickle.load(fp)
                self.dict_sample_id_to_num_frons[sample_name] = len(
                    pk_file['frontiers'])

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
                # print(f'class {v} is not a goal object.')

        # =========================== load pre-built semantic and occupancy maps ========================
        print('load maps ...')
        sem_map_npy = np.load(
            f'output/semantic_map/{self.split}/{self.scene_name}_{self.floor_id}/BEV_semantic_map.npy',
            allow_pickle=True).item()
        gt_sem_map, pose_range, coords_range, WH = read_map_npy(
            sem_map_npy)
        occ_map_npy = np.load(
            f'output/semantic_map/{self.split}/{self.scene_name}_{self.floor_id}/BEV_occupancy_map.npy',
            allow_pickle=True).item()
        gt_occ_map, _, _, _ = read_occ_map_npy(occ_map_npy)
        gt_occupancy_map = gt_occ_map
        gt_occupancy_map = np.where(gt_occupancy_map == 1,
                                    cfg.FE.FREE_VAL, gt_occupancy_map)  # free cell
        self.gt_occupancy_map = np.where(
            gt_occupancy_map == 0, cfg.FE.COLLISION_VAL, gt_occupancy_map)  # occupied cell

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

        # ================== colorize the semantic map and merge with occupancy map ==================
        self.color_semantic_map = apply_color_to_map(
            semantic_occupancy_map, type_categories='LVIS')
        enlarged_occ_map = cv2.resize(
            gt_occ_map, (self.W * cfg.SEM_MAP.VIS_ENLARGE_RATIO, self.H * cfg.SEM_MAP.VIS_ENLARGE_RATIO), interpolation=cv2.INTER_NEAREST)
        # turn free space into white
        self.color_semantic_map[enlarged_occ_map > 0] = np.ones(3) * 255

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
        self.vis_list_instances = []
        for idx_ins in range(1, num_ins + 1):
            mask_ins = (self.instance_label == idx_ins)
            if np.sum(mask_ins) > 50:  # should have at least 50 pixels
                #print(f'idx_ins = {idx_ins}')
                x_coords = xv[mask_ins]
                y_coords = yv[mask_ins]
                ins_center = (floor(np.median(x_coords)),
                              floor(np.median(y_coords)))
                ins_cat = semantic_occupancy_map[int(
                    y_coords[0]), int(x_coords[0])]
                if self.idx2cat_dict[ins_cat] in self.goal_category_set:
                    vis_ins = {}
                    vis_ins['center'] = ins_center
                    vis_ins['cat'] = ins_cat
                    self.vis_list_instances.append(vis_ins)
                    ins = {}
                    ins['center'] = (ins_center[0] // cfg.SEM_MAP.VIS_ENLARGE_RATIO,
                                     ins_center[1] // cfg.SEM_MAP.VIS_ENLARGE_RATIO)
                    ins['cat'] = ins_cat
                    ins['id_ins'] = idx_ins
                    self.list_instances.append(ins)

    def __len__(self):
        sum_frons = 0
        for k, v in self.dict_sample_id_to_num_frons.items():
            sum_frons += v
        return sum_frons

    def decide_sample_name_from_index(self, index):
        rest_num_frons = index
        for k, v in self.dict_sample_id_to_num_frons.items():
            if v > rest_num_frons:
                fron_idx = rest_num_frons
                return k, fron_idx
            else:
                rest_num_frons -= v

    def __getitem__(self, index):
        snapshot_idx, fron_idx = self.decide_sample_name_from_index(index)

        # ================= read the explored map ====================
        with bz2.BZ2File(f'{self.data_folder}/{self.scene_name}_{self.floor_id}/{snapshot_idx}.pbz2', 'rb') as fp:
            pk_file = cPickle.load(fp)
            cur_sem_map = pk_file['sseg_map']
            cur_occ_map = pk_file['occ_map']
            sampled_frontier = pk_file['frontiers'][fron_idx]

        # initialize a location
        goal_category = self.random.choice(
            list(self.goal_category_set))  # 'toy'
        goal_category_index = self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
            goal_category)]
        #goal_category = 'toy'
        print(f'goal_category = {goal_category}')

        # ============================ localize the object on the map ==================
        # find the instances under the same goal_category_idx
        ins_list_containing_goal = []
        for ins in self.list_instances:
            cat = self.idx2cat_dict[ins['cat']]
            cat_index = self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
                cat)]
            if cat_index == goal_category_index:
                #print(f'cat = {cat}')
                ins_list_containing_goal.append(ins)

        # ===========================================================================
        observed_area_flag = (cur_occ_map != cfg.FE.UNOBSERVED_VAL)

        #  expand the object center to reach the free space
        object_expand_occupancy_map = self.gt_occupancy_map.copy()

        # get the free space coordinates
        mask_free_cells = (self.gt_occupancy_map == cfg.FE.FREE_VAL)

        # create an entirely free map
        dist_occupancy_map = np.ones(
            self.gt_occupancy_map.shape).astype('int16')
        m = MCPG(dist_occupancy_map, fully_connected=True)

        for idx in range(len(ins_list_containing_goal)):
            starts = []
            for ins in [ins_list_containing_goal[idx]]:
                center = ins['center']
                starts.append([center[1], center[0]])

            cost_array, tracebacks_array = m.find_costs(starts)
            min_dist_to_free_cell = ceil(np.min(cost_array[mask_free_cells]))
            # expand from the ins centroid by min_dist
            mask_expand = (cost_array <= min_dist_to_free_cell)
            object_expand_occupancy_map[mask_expand] = cfg.FE.FREE_VAL

        # ===========================================================================

        free_but_unobserved_flag = np.logical_and(
            object_expand_occupancy_map == cfg.FE.FREE_VAL, observed_area_flag == False)
        free_but_unobserved_flag = scipy.ndimage.maximum_filter(
            free_but_unobserved_flag, size=3)

        labels, nb = scipy.ndimage.label(free_but_unobserved_flag)

        for ii in range(nb):
            component = (labels == (ii + 1))
            # for fron in frontiers:
            fron = sampled_frontier
            if component[fron['centroid'][0], fron['centroid'][1]]:
                starts = [[fron['centroid'][0], fron['centroid'][1]]]
                ends = []
                for ins in ins_list_containing_goal:
                    center = ins['center']
                    if component[center[1], center[0]]:
                        ends.append([center[1], center[0]])
                if len(ends) == 0:
                    dist = cfg.NAVI.MAXIMUM_DIST_TO_OBJ_GOAL
                else:
                    component_distance_map = component.copy()
                    component_distance_map[component_distance_map == 0] = 1000
                    comp_m = MCPG(component_distance_map,
                                  fully_connected=True)
                    cost_array, _ = comp_m.find_costs(
                        starts, ends)
                    ends_coords = tuple(np.array(ends).T.tolist())
                    ends_costs = cost_array[ends_coords]
                    dist = min(ends_costs)
                    nearest_end_idx = np.argmin(ends_costs)

                print(f'dist = {dist:.2f}')

                if False:  # cfg.NAVI.FLAG_VISUALIZE_FRONTIER_POTENTIAL:
                    fig, ax = plt.subplots(nrows=2,
                                           ncols=2,
                                           figsize=(12, 7))

                    ax[0, 0].imshow(self.color_semantic_map)

                    x, y = [], []
                    for ins in self.vis_list_instances:
                        center = ins['center']
                        cat = ins['cat']

                        x.append(center[0])
                        y.append(center[1])

                        try:
                            cat_name = self.idx2cat_dict[cat]
                        except:
                            cat_name = 'unknown'
                        ax[0, 0].text(center[0], center[1], cat_name)

                    ax[0, 0].scatter(x=x, y=y, c='b', s=5)
                    ax[0, 0].get_xaxis().set_visible(False)
                    ax[0, 0].get_yaxis().set_visible(False)
                    ax[0, 0].set_title('semantic map')

                    ax[0, 1].imshow(self.gt_occupancy_map, cmap='gray')

                    x, y = [], []
                    for ins in ins_list_containing_goal:
                        center = ins['center']
                        cat = ins['cat']

                        x.append(center[0])
                        y.append(center[1])

                        try:
                            cat_name = self.idx2cat_dict[cat]
                        except:
                            cat_name = 'unknown'
                        ax[0, 1].text(center[0], center[1],
                                      cat_name, c='green')

                    ax[0, 1].scatter(x=x, y=y, marker='*', c='yellow', s=5)
                    ax[0, 1].get_xaxis().set_visible(False)
                    ax[0, 1].get_yaxis().set_visible(False)
                    ax[0, 1].set_title('goal object centroids')

                    ax[1, 0].imshow(cur_occ_map, cmap='gray')
                    ax[1, 0].scatter(fron['points'][1],
                                     fron['points'][0],
                                     c='yellow',
                                     zorder=2)
                    ax[1, 0].scatter(fron['centroid'][1],
                                     fron['centroid'][0],
                                     c='red',
                                     zorder=2)
                    ax[1, 0].get_xaxis().set_visible(False)
                    ax[1, 0].get_yaxis().set_visible(False)
                    ax[1, 0].set_title('explored occupancy map')

                    ax[1, 1].imshow(component, cmap='gray')
                    goal_x, goal_y = [], []
                    for ins in ins_list_containing_goal:
                        center = ins['center']
                        if component[center[1], center[0]]:
                            goal_x.append(center[0])
                            goal_y.append(center[1])
                    ax[1, 1].scatter(goal_x, goal_y,
                                     marker='*',
                                     s=50,
                                     c='cyan',
                                     zorder=5)
                    # visualize the trajectory
                    if dist < cfg.NAVI.MAXIMUM_DIST_TO_OBJ_GOAL:
                        traj = comp_m.traceback(ends[nearest_end_idx])
                        traj = np.array(traj)
                        ax[1, 1].scatter(traj[:, 1], traj[:, 0], s=10,
                                         c='blue', zorder=2)
                        ax[1, 1].scatter(fron['centroid'][1],
                                         fron['centroid'][0],
                                         s=50,
                                         c='red',
                                         zorder=4)
                    ax[1, 1].get_xaxis().set_visible(False)
                    ax[1, 1].get_yaxis().set_visible(False)
                    ax[1, 1].set_title(f'component {ii}')

                    fig.tight_layout()
                    #plt.title(f'component {ii}')
                    plt.show()

                # ======================= prepare the return =================
                tensor_rgb = torch.tensor(fron['rgb']).float().permute(2, 0, 1)
                assert tensor_rgb.shape[1] == cfg.SENSOR.OBS_WIDTH
                goal_obj = goal_category
                tensor_dist = torch.tensor([dist]).float()

                # compute the bbox
                sseg_img = fron['sseg']
                img_hm3d_cat_index_list = np.unique(sseg_img)
                bbox_list = []
                # create image coordinates
                coords = get_img_coordinates(sseg_img)
                # convert sseg_img into bbox
                for hm3d_cat_index in img_hm3d_cat_index_list:
                    lvis_cat_name = self.idx2cat_dict[hm3d_cat_index]
                    # if current category is in the goal category list
                    if lvis_cat_name in self.goal_category_set:
                        #print(f'cat_name = {lvis_cat_name}')
                        # create binary image for current category index
                        cat_binary_map = np.zeros(
                            sseg_img.shape, dtype=np.int16)
                        cat_binary_map[sseg_img == hm3d_cat_index] = 1
                        instance_label, num_ins = skimage.measure.label(
                            cat_binary_map, background=0, connectivity=1, return_num=True)
                        #print(f'num_ins = {num_ins}')
                        # plt.imshow(cat_binary_map)
                        # plt.show()
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
                            if x2 - x1 > 5 and y2 - y1 > 5:
                                bbox_list.append([x1, y1, x2, y2, cat_id])

                return {'rgb': tensor_rgb, 'bbox': bbox_list, 'goal_obj': goal_obj, 'dist': tensor_dist}


def get_all_view_dataset(split, data_folder, hm3d_to_lvis_dict, LVIS_dict):
    # read the scene folders
    scene_list = sorted(
        next(os.walk(f'{data_folder}/{split}'))[1])

    ds_list = []
    for scene_floor in scene_list:
        scene_name, floor_id = scene_floor.split('_')
        view_ds = view_dataset(split, scene_name, floor_id,
                               f'{data_folder}/{split}', hm3d_to_lvis_dict, LVIS_dict)
        ds_list.append(view_ds)

    concat_ds = data.ConcatDataset(ds_list)
    return concat_ds


def my_collate(batch):
    output_dict = {}

    batch_rgb = [dict['rgb'] for dict in batch]
    output_dict['rgb'] = torch.stack(batch_rgb, 0)

    batch_goal_obj = [dict['goal_obj'] for dict in batch]
    output_dict['goal_obj'] = batch_goal_obj

    batch_dist = [dict['dist'] for dict in batch]
    output_dict['dist'] = torch.stack(batch_dist, 0)

    batch_bbox = [dict['bbox'] for dict in batch]
    output_dict['bbox'] = batch_bbox

    return output_dict


if __name__ == "__main__":

    cfg.merge_from_file('configs/exp_train_input_view_model_resnet.yaml')
    cfg.freeze()

    split = 'train'
    scene_name = '00009-vLpv2VX547B'
    floor_ids = [0, 1]

    data_folder = 'output/training_data_input_view_1000samples/train'

    # =================== read the semantic map ===============
    hm3d_to_lvis_dict = np.load(
        'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

    # load the LVIS categories
    with bz2.BZ2File('output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
        LVIS_dict = cPickle.load(fp)
        # lvis_cat_list = LVIS_dict['categories']
        # lvis_rowid_to_catid_dict = LVIS_dict['rowid2catid_dict']
        # lvis_cat_embedding = LVIS_dict['embedding']

    ds_list = []
    for floor_id in floor_ids:
        view_ds = view_dataset(split, scene_name, floor_id,
                               data_folder, hm3d_to_lvis_dict, LVIS_dict)
        ds_list.append(view_ds)

    concat_ds = data.ConcatDataset(ds_list)

    for i in range(len(concat_ds)):
        a = concat_ds[i]
