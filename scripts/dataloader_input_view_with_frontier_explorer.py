import numpy as np
import matplotlib.pyplot as plt
from modeling.utils.baseline_utils import get_img_coordinates
import bz2
import _pickle as cPickle
from core import cfg
import skimage.measure
#from random import Random
import random
import torch.utils.data as data
import os
import glob
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def process_lvis_dict(hm3d_to_lvis_dict, LVIS_dict):
    goal_obj_list = sorted(list(set(hm3d_to_lvis_dict.values())))  # size: 351

    goal_obj_index_list = list(set(LVIS_dict['rowid2catid_dict'][LVIS_dict['cat_synonyms'].index(
        cat_syn)] for cat_syn in goal_obj_list))  # size: 310

    lvis_cat_name_to_lvis_id_dict = {cat_syn: LVIS_dict['rowid2catid_dict'][LVIS_dict['cat_synonyms'].index(
        cat_syn)] for cat_syn in goal_obj_list}

    lvis_id_to_lvis_cat_names_dict = {}
    for cat_syn in goal_obj_list:
        lvis_id = LVIS_dict['rowid2catid_dict'][LVIS_dict['cat_synonyms'].index(
            cat_syn)]
        if lvis_id in lvis_id_to_lvis_cat_names_dict:
            lvis_id_to_lvis_cat_names_dict[lvis_id].append(cat_syn)
        else:
            lvis_id_to_lvis_cat_names_dict[lvis_id] = [cat_syn]
    return goal_obj_list, goal_obj_index_list, lvis_cat_name_to_lvis_id_dict, lvis_id_to_lvis_cat_names_dict


class view_dataset(data.Dataset):

    def __init__(self, split, scene_name, floor_id=0, data_folder='', hm3d_to_lvis_dict=None,
                 LVIS_dict=None):
        #self.random = Random(cfg.GENERAL.RANDOM_SEED)

        self.split = split
        self.scene_name = scene_name
        self.floor_id = floor_id

        self.data_folder = data_folder

        self.hm3d_to_lvis_dict = hm3d_to_lvis_dict
        self.LVIS_dict = LVIS_dict

        # goal_obj_list = sorted(
        #     list(set(hm3d_to_lvis_dict.values())))  # size: 351
        # self.goal_obj_index_list = list(set(self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
        #     cat_syn)] for cat_syn in goal_obj_list))  # size: 310

        self.goal_obj_list, self.goal_obj_index_list, self.lvis_cat_name_to_lvis_id_dict, self.lvis_id_to_lvis_cat_names_dict = process_lvis_dict(
            hm3d_to_lvis_dict, LVIS_dict)

        print(
            f'start dataset on scene {self.scene_name} floor {self.floor_id} =>')

        # ============================= build the dict of views ===============================
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

        self.preprocess = _transform(224)

    def __len__(self):
        return len(self.sample_name_list)

    def __getitem__(self, index):
        sample_name = self.sample_name_list[index]

        # ================= read the explored map ====================
        with bz2.BZ2File(f'{self.data_folder}/{self.scene_name}_{self.floor_id}/{sample_name}.pbz2', 'rb') as fp:
            fron = cPickle.load(fp)

        # ================ deal with imbalance =================
        # mat_dist = fron['mat_dist_to_cat']
        # mask_reachable = mat_dist < cfg.NAVI.MAXIMUM_DIST_TO_OBJ_GOAL
        # mat_dist[mask_reachable] = 1
        # mat_dist[~mask_reachable] = 0

        # goal_category = random.choice(
        #     list(self.goal_category_set))  # 'toy'
        # goal_category_index = self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
        #     goal_category)]
        #goal_category = 'toy'
        #print(f'goal_category = {goal_category}')

        # ======================= prepare the return =================
        # tensor_rgb = torch.tensor(fron['rgb']).float().permute(2, 0, 1)
        # assert tensor_rgb.shape[1] == cfg.SENSOR.OBS_WIDTH
        rgb_img = Image.fromarray(fron['rgb'])
        tensor_rgb = self.preprocess(rgb_img)
        #goal_obj = goal_category
        goal_obj = []
        for goal_obj_index in self.goal_obj_index_list:
            synonyms = self.lvis_id_to_lvis_cat_names_dict[goal_obj_index]
            goal_obj.append(random.choice(synonyms))

        # compute dist
        mat_dist = fron['mat_dist_to_cat']
        # dist = mat_dist[self.goal_obj_index_list.index(
        #     goal_category_index)]
        # if dist > cfg.NAVI.MAXIMUM_DIST_TO_OBJ_GOAL - 1:
        #     dist = 0
        # else:
        #     dist = 1
        dist = (mat_dist < cfg.NAVI.MAXIMUM_DIST_TO_OBJ_GOAL)
        #print(f'dist = {dist}')
        tensor_dist = torch.tensor(dist).long()
        #print(f'tensor_dist = {tensor_dist}')

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

        return {'rgb': tensor_rgb, 'bbox': bbox_list, 'goal_obj': goal_obj, 'dist': tensor_dist,
                'original_img': rgb_img}


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

    batch_original_img = [dict['original_img'] for dict in batch]
    output_dict['original_img'] = batch_original_img

    return output_dict


if __name__ == "__main__":

    cfg.merge_from_file('configs/exp_train_input_view_model_resnet.yaml')
    cfg.freeze()

    split = 'train'
    scene_name = '00009-vLpv2VX547B'
    floor_ids = [0, 1]

    data_folder = cfg.PRED.VIEW.PROCESSED_VIEW_SAVED_FOLDER

    # =================== read the semantic map ===============
    hm3d_to_lvis_dict = np.load(
        'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

    # load the LVIS categories
    with bz2.BZ2File('output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
        LVIS_dict = cPickle.load(fp)
        # lvis_cat_list = LVIS_dict['categories']
        # lvis_rowid_to_catid_dict = LVIS_dict['rowid2catid_dict']
        # lvis_cat_embedding = LVIS_dict['embedding']

    concat_ds = get_all_view_dataset(
        'train', data_folder, hm3d_to_lvis_dict, LVIS_dict)

    for i in range(len(concat_ds)):
        a = concat_ds[i]
