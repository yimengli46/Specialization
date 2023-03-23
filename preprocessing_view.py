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
import os
import glob
import argparse
import multiprocessing
from matplotlib.gridspec import GridSpec


class preprocessing_view():

    def __init__(self, split, scene_name, floor_id=0, data_folder='', saved_folder='', hm3d_to_lvis_dict=None,
                 LVIS_dict=None):
        self.random = Random(cfg.GENERAL.RANDOM_SEED)

        self.split = split
        self.scene_name = scene_name
        self.floor_id = floor_id

        self.data_folder = data_folder

        # ============= create scene folder =============
        scene_folder = f'{saved_folder}/{self.scene_name}_{self.floor_id}'
        if not os.path.exists(scene_folder):
            os.mkdir(scene_folder)
        self.scene_folder = scene_folder

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

        # ============================= build the dict of views ===============================
        print('build the scene view dictionary.')
        self.sample_name_list = [os.path.splitext(os.path.basename(x))[0]
                                 for x in sorted(glob.glob(f'{data_folder}/{self.scene_name}_{self.floor_id}/*.pbz2'))]
        print(f'find {len(self.sample_name_list)} files.')

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
            gt_occ_map, (self.W * cfg.SEM_MAP.VIS_ENLARGE_RATIO,
                         self.H * cfg.SEM_MAP.VIS_ENLARGE_RATIO), interpolation=cv2.INTER_NEAREST)
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

    def write_to_file(self):
        count_view = 0
        for sample_name in self.sample_name_list:
            print(f'sample_name = {sample_name}')

            # ================= read the explored map ====================
            with bz2.BZ2File(f'{self.data_folder}/{self.scene_name}_{self.floor_id}/{sample_name}.pbz2', 'rb') as fp:
                pk_file = cPickle.load(fp)
                cur_sem_map = pk_file['sseg_map']
                cur_occ_map = pk_file['occ_map']
                frontiers = pk_file['frontiers']

            # some preparation work
            observed_area_flag = (cur_occ_map != cfg.FE.UNOBSERVED_VAL)
            # get the free space coordinates
            mask_free_cells = (self.gt_occupancy_map == cfg.FE.FREE_VAL)

            # initialize a matrix to save the view to each class
            mat_dist_view_to_cat = np.ones(
                (len(frontiers), self.num_cats), dtype=np.float32) * cfg.NAVI.MAXIMUM_DIST_TO_OBJ_GOAL

            # initialize a location
            for goal_category in list(self.goal_category_set):  # 'toy'
                goal_category_index = self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
                    goal_category)]
                #print(f'goal_category = {goal_category}')

                # ============================ localize the object on the map ==================
                # find the instances containing the goal category
                ins_list_containing_goal = []
                for ins in self.list_instances:
                    cat = self.idx2cat_dict[ins['cat']]
                    cat_index = self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
                        cat)]
                    if cat_index == goal_category_index:
                        #print(f'cat = {cat}')
                        ins_list_containing_goal.append(ins)

                # ===========================================================================
                #  expand the object center to reach the free space
                object_expand_occupancy_map = self.gt_occupancy_map.copy()

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
                    min_dist_to_free_cell = ceil(
                        np.min(cost_array[mask_free_cells]))
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
                    for fron_idx, fron in enumerate(frontiers):
                        if component[fron['centroid'][0], fron['centroid'][1]]:
                            starts = [
                                [fron['centroid'][0], fron['centroid'][1]]]
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

                            mat_dist_view_to_cat[fron_idx,
                                                 self.goal_obj_index_list.index(goal_category_index)] = dist

                            #print(f'dist = {dist:.2f}')

                            # visualize a single view and distance to a goal
                            if False:
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
                                    ax[0, 0].text(
                                        center[0], center[1], cat_name)

                                ax[0, 0].scatter(x=x, y=y, c='b', s=5)
                                ax[0, 0].get_xaxis().set_visible(False)
                                ax[0, 0].get_yaxis().set_visible(False)
                                ax[0, 0].set_title('semantic map')

                                ax[0, 1].imshow(
                                    self.gt_occupancy_map, cmap='gray')

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

                                ax[0, 1].scatter(
                                    x=x, y=y, marker='*', c='yellow', s=5)
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
                                    traj = comp_m.traceback(
                                        ends[nearest_end_idx])
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

            # =========================== visualize the matrices with the frontiers ===================
            if False:
                fig = plt.figure(layout="tight", figsize=(12, 7))
                gs = GridSpec(2, 2, figure=fig)

                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, :])

                ax1.imshow(self.color_semantic_map)

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
                    ax1.text(
                        center[0], center[1], cat_name)

                ax1.scatter(x=x, y=y, c='b', s=5)
                ax1.get_xaxis().set_visible(False)
                ax1.get_yaxis().set_visible(False)
                ax1.set_title('semantic map')

                ax2.imshow(cur_occ_map, cmap='gray')
                for fron in frontiers:
                    ax2.scatter(fron['points'][1],
                                fron['points'][0],
                                c='green',
                                zorder=2)
                    ax2.scatter(fron['centroid'][1],
                                fron['centroid'][0],
                                c='red',
                                zorder=2)
                ax2.get_xaxis().set_visible(False)
                ax2.get_yaxis().set_visible(False)
                ax2.set_title('explored occupancy map')

                vis_mat_dist = mat_dist_view_to_cat.copy()
                vis_mat_dist[mat_dist_view_to_cat > 9999] = 0
                ax3.imshow(vis_mat_dist, vmin=0.0, vmax=100.0)
                # ax3.get_xaxis().set_visible(False)
                # ax3.get_yaxis().set_visible(False)
                ax3.set_title(
                    'distance between frontiers and all categories')

                fig.tight_layout()
                #plt.title(f'component {ii}')
                plt.show()

            # ============== write frontier result to disk ================
            # write fron, mat[fron_idx] to a file
            for fron_idx, fron_data in enumerate(frontiers):
                fron_data['mat_dist_to_cat'] = mat_dist_view_to_cat[fron_idx]
                #print(f'mat.shape = {mat_dist_view_to_cat[fron_idx].shape}')

                sample_name = str(count_view).zfill(5)
                with bz2.BZ2File(f'{self.scene_folder}/{sample_name}.pbz2', 'w') as fp:
                    cPickle.dump(
                        fron_data,
                        fp
                    )
                # '''

                count_view += 1


def multi_run_wrapper(args):
    """ wrapper for multiprocessor """
    preprocess_view = preprocessing_view(
        args[0], args[1], args[2], args[3], args[4], args[5], args[6])
    preprocess_view.write_to_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True, default='val')
    args = parser.parse_args()
    cfg.merge_from_file('configs/exp_preprocessing_view_dataset.yaml')
    cfg.freeze()

    split = args.split

    gen_samples_saved_folder = cfg.PRED.VIEW.GEN_SAMPLES_SAVED_FOLDER
    data_folder = f'{gen_samples_saved_folder}/{split}'

    # ========================= create folders ===================
    output_folder = cfg.PRED.VIEW.PROCESSED_VIEW_SAVED_FOLDER
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    saved_folder = f'{output_folder}/{split}'
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)

    # ===================== load the existing generated-samples folders ================
    gen_samples_folder_list = sorted(
        next(os.walk(f'{data_folder}'))[1])
    list_scene_floor_tuple = []
    for gen_samples_folder in gen_samples_folder_list:
        scene_name, floor_id = gen_samples_folder.split('_')
        list_scene_floor_tuple.append((scene_name, floor_id))

    hm3d_to_lvis_dict = np.load(
        'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

    # load the LVIS categories
    with bz2.BZ2File(f'output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
        LVIS_dict = cPickle.load(fp)

    # ==================== process each scene ========================
    if cfg.PRED.VIEW.multiprocessing == 'single':  # single process
        for scene_floor_tuple in list_scene_floor_tuple:
            scene_name, floor_id = scene_floor_tuple
            preprocess_view = preprocessing_view(
                split, scene_name, floor_id, data_folder, saved_folder, hm3d_to_lvis_dict, LVIS_dict)
            preprocess_view.write_to_file()
    elif cfg.PRED.VIEW.multiprocessing == 'mp':
        with multiprocessing.Pool(processes=cfg.PRED.VIEW.NUM_PROCESS) as pool:
            args1, args2 = [], []
            for scene_floor_tuple in list_scene_floor_tuple:
                scene_name, floor_id = scene_floor_tuple
                args1.append(scene_name)
                args2.append(floor_id)
            args0 = [split for _ in range(len(args1))]
            args3 = [data_folder for _ in range(len(args1))]
            args4 = [saved_folder for _ in range(len(args1))]
            args5 = [hm3d_to_lvis_dict for _ in range(len(args1))]
            args6 = [LVIS_dict for _ in range(len(args1))]
            pool.map(multi_run_wrapper, list(
                zip(args0, args1, args2, args3, args4, args5, args6)))
            pool.close()
