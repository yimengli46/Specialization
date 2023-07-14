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

        self.room_types = ['a living room', 'a bathroom', 'a dining room', 'a kitchen',
                           'a bedroom', 'a pantry', 'an office', 'a garage', 'outdoor', 'a corridor']

        self.common_objs = [19, 232, 464, 558, 889, 285, 367, 610, 831, 1050, 181, 438, 351, 982, 77, 170, 390,
                            135, 837, 1139, 609, 961, 1097, 468, 710, 1013, 1018, 1108, 1077, 957, 955, 68, 444, 677]
        self.common_objs_row_ids = [self.goal_obj_index_list.index(i) for i in self.common_objs]

        self.targets = {
            'chair': [19, 232, 464, 558, 889],
            'sofa': [982],
            'plant': [135, 837, 1139],
            'bed': [77, 170],
            'toilet': [1097],
            'tv': [1077],
        }

        self.weighted_co_matrix_obj_and_obj = np.load(
            'output/weighted_kg/weighted_co_matrix_obj_and_obj_train_all.npy', allow_pickle=True)
        self.weighted_co_matrix_room_and_obj = np.load(
            'output/weighted_kg/weighted_co_matrix_room_and_obj_train_all.npy', allow_pickle=True)

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

        # ====================== prepare input for MLP =====================
        map_dist_to_cat = fron['map_dist_to_cat']

        # create input tensor with shape num_targets x 10
        # create output tensor with shape num_targets
        num_targets = len(self.targets.keys())
        num_features = 2 * len(self.common_objs_row_ids) + 2 * len(self.room_types)
        input_tensor = torch.zeros((num_targets, num_features)).float()
        output_tensor = torch.zeros(num_targets)

        # check the common objects in the view
        view_common_obj_row_ids = set(np.where(map_dist_to_cat == 1)[1]).intersection(self.common_objs_row_ids)

        # load the room types
        room_type_preds = np.load(
            f'output/CLIP_room_type/{self.split}/{self.scene_name}_{self.floor_id}/{sample_name}_clip_room_types.npy', allow_pickle=True)

        for idx_target, goal_obj in enumerate(list(self.targets.keys())):
            # print(f'goal_obj = {goal_obj}')
            goal_obj_row_ids = [self.goal_obj_index_list.index(i) for i in self.targets[goal_obj]]

            for common_obj_row_id in view_common_obj_row_ids:
                cooccur_value = self.weighted_co_matrix_obj_and_obj[goal_obj_row_ids, common_obj_row_id].max()
                i_common_obj_row_id = self.common_objs_row_ids.index(
                    common_obj_row_id)  # index of current obj_row_id in all the objects
                input_tensor[idx_target, i_common_obj_row_id * 2] = 1
                input_tensor[idx_target, i_common_obj_row_id * 2 + 1] = float(cooccur_value)

            for idx_room in range(len(self.room_types)):
                cooccur_value = self.weighted_co_matrix_room_and_obj[idx_room, goal_obj_row_ids].max()
                input_tensor[idx_target, 2 * len(self.common_objs_row_ids) + 2 *
                             idx_room] = float(room_type_preds[idx_room])
                input_tensor[idx_target, 2 * len(self.common_objs_row_ids) + 2 *
                             idx_room + 1] = float(cooccur_value)

        # ======================= prepare other return =================
        rgb_img = Image.fromarray(fron['rgb'])
        tensor_rgb = self.preprocess(rgb_img)

        goal_obj = []
        for goal_obj_index in self.goal_obj_index_list:
            synonyms = self.lvis_id_to_lvis_cat_names_dict[goal_obj_index]
            goal_obj.append(random.choice(synonyms))

        # compute dist
        mat_dist = fron['map_dist_to_cat']
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

        for idx_target, goal_obj in enumerate(list(self.targets.keys())):
            # print(f'goal_obj = {goal_obj}')
            goal_obj_row_ids = [self.goal_obj_index_list.index(i) for i in self.targets[goal_obj]]
            target_dist = mat_dist[0, goal_obj_row_ids].max()
            output_tensor[idx_target] = target_dist
        output_tensor = output_tensor.long()

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

        return {'rgb': tensor_rgb, 'bbox': bbox_list, 'goal_obj': goal_obj, 'dist': tensor_dist,
                'original_img': rgb_img, 'original_dist': gt_mat_dist, 'input': input_tensor, 'output': output_tensor}


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

    batch_input = [dict['input'] for dict in batch]
    output_dict['input'] = torch.stack(batch_input, 0)

    batch_output = [dict['output'] for dict in batch]
    output_dict['output'] = torch.stack(batch_output, 0)

    return output_dict


if __name__ == "__main__":

    cfg.merge_from_file(
        'configs/exp_train_input_view_model_MLP.yaml')
    cfg.freeze()

    split = 'train'

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
        split, data_folder, hm3d_to_lvis_dict, LVIS_dict, test_transform, 'gt')

    for i in range(len(concat_ds)):
        print(f'i = {i}')
        a = concat_ds[i]
        # assert 1 == 2
