import numpy as np
import matplotlib.pyplot as plt
from modeling.utils.baseline_utils import get_img_coordinates
import bz2
import _pickle as cPickle
from core import cfg
import skimage.measure
# from random import Random
import random
import torch.utils.data as data
import os
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


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
                 LVIS_dict=None, transform=None, bbox_type='gt'):
        # self.random = Random(cfg.GENERAL.RANDOM_SEED)

        self.split = split
        self.scene_name = scene_name
        self.floor_id = floor_id

        self.data_folder = data_folder
        self.bbox_type = bbox_type

        if self.bbox_type == 'Detic':
            self.detection_folder = 'output/Detic_detections_of_input_view_by_densely_sample_locations'

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

        self.preprocess = transform

    def __len__(self):
        return len(self.sample_name_list)

    def __getitem__(self, index):
        sample_name = self.sample_name_list[index]
        # print(f'scene_name = {self.scene_name}, sample_name = {sample_name}')

        # for finding obj and dist mis-alignment images
        '''
        temp_str = f'{self.data_folder}/{self.scene_name}_{self.floor_id}/{sample_name}'
        return temp_str
        '''
        # ================= read the explored map ====================
        with bz2.BZ2File(f'{self.data_folder}/{self.scene_name}_{self.floor_id}/{sample_name}.pbz2', 'rb') as fp:
            fron = cPickle.load(fp)

        # ======================= prepare the return =================
        rgb_img = Image.fromarray(fron['rgb'])
        tensor_rgb = self.preprocess(rgb_img)
        # goal_obj = goal_category
        goal_obj = []
        for goal_obj_index in self.goal_obj_index_list:
            synonyms = self.lvis_id_to_lvis_cat_names_dict[goal_obj_index]
            goal_obj.append(random.choice(synonyms))

        # compute dist
        mat_dist = fron['mat_dist_to_cat']
        gt_mat_dist = mat_dist.copy()
        if cfg.PRED.VIEW.MULTILABEL_MODE == 'detected_only':
            mask = mat_dist > 1
            mat_dist[mask] = 0
        elif cfg.PRED.VIEW.MULTILABEL_MODE == 'detected_and_nearby':
            mask = mat_dist > 1
            mat_dist[mask] = 1
        # print(f'dist = {mat_dist}')
        tensor_dist = torch.tensor(mat_dist).long()
        # print(f'tensor_dist = {tensor_dist.shape}')

        # compute the bbox
        detector_pred = np.zeros((1, 310))
        if self.bbox_type == 'gt':
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
                    # print(f'cat_name = {lvis_cat_name}')
                    # create binary image for current category index
                    cat_binary_map = np.zeros(
                        sseg_img.shape, dtype=np.int16)
                    cat_binary_map[sseg_img == hm3d_cat_index] = 1
                    instance_label, num_ins = skimage.measure.label(
                        cat_binary_map, background=0, connectivity=1, return_num=True)
                    # print(f'num_ins = {num_ins}')
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
                            ind = self.goal_obj_index_list.index(cat_id)
                            detector_pred[0, ind] = 1
        elif self.bbox_type == 'Detic':
            with bz2.BZ2File(f'{self.detection_folder}/{self.split}/{self.scene_name}_{self.floor_id}/{sample_name}_Detic.pbz2', 'rb') as fp:
                pred_dict = cPickle.load(fp)
                num_instances = pred_dict['num_instances']
                pred_boxes = pred_dict['pred_boxes']
                scores = pred_dict['scores']
                pred_classes = pred_dict['pred_classes']

            bbox_list = []
            # go through from smallest conf to largest conf
            for instance_idx in range(num_instances - 1, -1, -1):
                cat_id = self.LVIS_dict['rowid2catid_dict'][self.LVIS_dict['cat_synonyms'].index(
                    self.goal_obj_list[pred_classes[instance_idx]])]
                # print(
                #     f'pred_classes = {self.goal_obj_list[pred_classes[instance_idx]]}')
                # print(f'cat_id = {cat_id}')
                bbox = pred_boxes[instance_idx]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                bbox_list.append([x1, y1, x2, y2, cat_id])
                ind = self.goal_obj_index_list.index(cat_id)
                detector_pred[0, ind] = scores[instance_idx]
        else:
            raise NotImplementedError(
                f"bbox type {self.bbox_type} not implemented.")

        return {'rgb': tensor_rgb, 'bbox': bbox_list, 'goal_obj': goal_obj, 'dist': tensor_dist,
                'original_img': rgb_img, 'original_dist': gt_mat_dist, 'detector_pred': detector_pred}


def get_all_view_dataset(split, data_folder, hm3d_to_lvis_dict, LVIS_dict, transforms=None, bbox_type='gt'):
    # read the scene folders
    scene_list = sorted(
        next(os.walk(f'{data_folder}/{split}'))[1])

    ds_list = []
    for scene_floor in scene_list:
        scene_name, floor_id = scene_floor.split('_')
        view_ds = view_dataset(split, scene_name, floor_id,
                               f'{data_folder}/{split}', hm3d_to_lvis_dict, LVIS_dict, transforms, bbox_type)
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
    output_dict['dist'] = torch.cat(batch_dist, 0)

    batch_bbox = [dict['bbox'] for dict in batch]
    output_dict['bbox'] = batch_bbox

    batch_original_img = [dict['original_img'] for dict in batch]
    output_dict['original_img'] = batch_original_img

    batch_original_dist = [dict['original_dist'] for dict in batch]
    output_dict['original_dist'] = np.concatenate(batch_original_dist, axis=0)

    batch_detector_pred = [dict['detector_pred'] for dict in batch]
    output_dict['detector_pred'] = np.concatenate(batch_detector_pred, axis=0)

    return output_dict


if __name__ == "__main__":

    cfg.merge_from_file(
        'configs/exp_train_input_view_multilabel_model_context_matrix.yaml')
    cfg.freeze()

    split = 'val'

    data_folder = cfg.PRED.VIEW.DENSELY_SAMPLED_LOCATIONS_SAVED_FOLDER

    # =================== read the semantic map ===============
    hm3d_to_lvis_dict = np.load(
        'output/knowledge_graph/hm3d_to_lvis_dict.npy', allow_pickle=True).item()

    # load the LVIS categories
    with bz2.BZ2File('output/knowledge_graph/LVIS_categories_and_embedding.pbz2', 'rb') as fp:
        LVIS_dict = cPickle.load(fp)
        # lvis_cat_list = LVIS_dict['categories']
        # lvis_rowid_to_catid_dict = LVIS_dict['rowid2catid_dict']
        # lvis_cat_embedding = LVIS_dict['embedding']

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    concat_ds = get_all_view_dataset(
        split, data_folder, hm3d_to_lvis_dict, LVIS_dict, test_transform, 'Detic')

    for i in range(len(concat_ds)):
        print(f'i = {i}')
        a = concat_ds[i]
