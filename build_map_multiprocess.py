import numpy as np
from modeling.utils.baseline_utils import create_folder
import habitat
from core import cfg
import argparse
import multiprocessing
import os
import json
from build_semantic_BEV_map_large_scale import build_sem_map
#from build_occ_map_from_densely_checking_cells_large_scale import build_occ_map


def build_env(split, scene_with_index, device_id=0):
    # ================================ load habitat env============================================
    config = habitat.get_config(
        config_paths=cfg.GENERAL.BUILD_MAP_CONFIG_PATH)
    config.defrost()
    config.SIMULATOR.SCENE = f'data/scene_datasets/hm3d/{split}/{scene_with_index}/{scene_with_index[6:]}.glb'
    config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
    config.SIMULATOR.SCENE_DATASET = 'data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = device_id
    config.freeze()
    env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
    return env


def build_floor(scene_with_index, output_folder, scene_floor_dict):
    # ============================ get a gpu
    device_id = gpu_Q.get()

    # ================ initialize habitat env =================
    env = build_env(scene_with_index, device_id=device_id)
    env.reset()

    scene_dict = scene_floor_dict[scene_with_index]
    for floor_id in list(scene_dict.keys()):
        height = scene_dict[floor_id]['y']
        scene_name = f'{scene_with_index}_{floor_id}'

        scene_output_folder = f'{output_folder}/{scene_name}'
        create_folder(scene_output_folder)

        build_sem_map(env, scene_output_folder, height)
        #build_occ_map(env, scene_output_folder, height, scene_name)

    env.close()

    # ================================ release the gpu============================
    gpu_Q.put(device_id)


def multi_run_wrapper(args):
    """ wrapper for multiprocessor """
    build_floor(args[0], args[1], args[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        required=False,
                        default='exp_specialization.yaml')
    args = parser.parse_args()

    cfg.merge_from_file(f'configs/{args.config}')
    cfg.freeze()

    # ====================== get the available GPU devices ============================
    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    devices = [int(dev) for dev in visible_devices]

    for device_id in devices:
        for _ in range(cfg.MP.PROC_PER_GPU):
            gpu_Q.put(device_id)

    # =============================== basic setup =======================================
    split = 'val'
    scene_floor_dict = np.load(
        f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy',
        allow_pickle=True).item()

    output_folder = 'output/semantic_map/{split}'
    create_folder(output_folder)

    # =================================analyze json file to get the semantic files =============================
    sem_filenames = []
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
            sem_filenames.append(sem_filename)

    args0 = sem_filenames
    with multiprocessing.Pool(processes=len(args0)) as pool:
        args1 = [output_folder for _ in range(len(args0))]
        args2 = [scene_floor_dict for _ in range(len(args0))]
        pool.map(multi_run_wrapper, list(zip(args0, args1, args2)))
        pool.close()
        pool.join()


if __name__ == "__main__":
    gpu_Q = multiprocessing.Queue()
    main()
