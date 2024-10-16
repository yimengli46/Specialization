import numpy as np
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
import gzip
import json
import quaternion as qt
from sklearn.mixture import GaussianMixture
from scipy import stats as st

split = 'train'  # 'val', 'train'
saved_folder = 'output/scene_height_distribution'

scene_start_y_dict = {}
scene_height_dict = {}

pointnav_foldername = f'data/datasets/pointnav_hm3d_v1/{split}/content'


def confirm_nComponents(X, top=1):
    bics = []
    min_bic = 0
    counter = 1
    # at most 3 floors is considered
    maximum_nComponents = min(3, top)
    print(f'max_nC = {maximum_nComponents}')
    opt_bic = 1
    # test the AIC/BIC metric between 1 and 10 components
    for i in range(1, maximum_nComponents + 1):
        gmm = GaussianMixture(n_components=counter, max_iter=1000,
                              random_state=0, covariance_type='full').fit(X)
        bic = gmm.bic(X)
        bics.append(bic)
        if bic < min_bic or min_bic == 0:
            min_bic = bic
            opt_bic = counter

        counter += 1
    return opt_bic


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
        second_slash = json_dir.find('/', first_slash + 1)

        sem_filename = json_dir[first_slash + 1:second_slash]
        point_filename = json_dir[first_slash + 7:second_slash]

        point_filenames.append(point_filename)
        sem_filenames.append(sem_filename)

# ======================================== analyze pointgoal nav episodes =================================
for filename in point_filenames:
    print(f'filename is {filename}.')
    with gzip.open(f'{pointnav_foldername}/{filename}.json.gz', 'rb') as f:
        data = json.loads(f.read())
        episodes = data['episodes']

        # ============================= collect the start point y of each scene =========================
        for episode in episodes:
            scene_id = episode['scene_id']

            pos_slash = scene_id.rfind('/')
            pos_dot = scene_id.rfind('.')-6
            episode_scene = scene_id[pos_slash+1:pos_dot]

            start_pose_y = episode['start_position'][1]

            if episode_scene in scene_start_y_dict:
                scene_start_y_dict[episode_scene].append(start_pose_y)
            else:
                scene_start_y_dict[episode_scene] = [start_pose_y]

# ================================= decide number of floors of each scene ===========================
scene_floor_dict = {}
for idx_scene, scene_with_index in enumerate(sem_filenames):
    height_lst = scene_start_y_dict[scene_with_index[6:]]
    values, counts = np.unique(height_lst, return_counts=True)

    num_floors = confirm_nComponents(
        np.array(height_lst).reshape(-1, 1), top=len(counts))
    print(f'num_floors = {num_floors}')
    print(f'values = {values}, counts = {counts}')

    gm = GaussianMixture(n_components=num_floors).fit(
        np.array(height_lst).reshape(-1, 1))
    labels = gm.predict(np.array(height_lst).reshape(-1, 1))
    sorted_weight_idx = np.argsort(np.array(gm.weights_))[::-1]
    # compute the mode as the peak_heights instead of the mean
    #peak_heights = gm.means_[sorted_weight_idx]
    peak_heights = []
    for weight_idx in sorted_weight_idx:
        heights_in_this_component = np.array(height_lst)[labels == weight_idx]
        mode_heights = st.mode(heights_in_this_component)
        if mode_heights[0].shape[0] > 1:
            mode_height = list(mode_heights[0])[0]
        else:
            mode_height = mode_heights[0].item()
        print(f'mode_heights = {mode_heights}, mode_height = {mode_height}')
        peak_heights.append(mode_height)

    # ================================ summarize the y values of each scene =========================
    scene_floor_dict[scene_with_index] = {}

    count_floor = 0
    for idx_height, height in enumerate(peak_heights):
        # check if within 1.5m of previous heights
        flag = False
        for i in range(0, idx_height):
            prev_height = peak_heights[i]
            if abs(prev_height - height) <= 1.5:
                flag = True

        if not flag:
            scene_floor_dict[scene_with_index][count_floor] = {}
            scene_floor_dict[scene_with_index][count_floor]['y'] = height
            count_floor += 1

    print(f'{scene_floor_dict[scene_with_index]}')

    print(f'------------------------------------------------')


# ===============================assign episodes to each floor =============================
gap_thresh = 0.01
for scene_with_index in sem_filenames:
    print(f'filename is {scene_with_index}.')
    with gzip.open(f'{pointnav_foldername}/{scene_with_index[6:]}.json.gz', 'rb') as f:
        data = json.loads(f.read())
        episodes = data['episodes']

        for episode in episodes:
            scene_id = episode['scene_id']

            pos_slash = scene_id.rfind('/')
            pos_dot = scene_id.rfind('.')-6

            start_pose_y = episode['start_position'][1]

            for idx_floor in list(scene_floor_dict[scene_with_index].keys()):
                floor_y = scene_floor_dict[scene_with_index][idx_floor]['y']
                if abs(start_pose_y - floor_y) < gap_thresh:
                    x = episode['start_position'][0]
                    z = episode['start_position'][2]

                    a, b, c, d = episode['start_rotation']
                    agent_rot = qt.quaternion(a, b, c, d)
                    heading_vector = quaternion_rotate_vector(
                        agent_rot.inverse(), np.array([0, 0, -1]))
                    phi = round(
                        cartesian_to_polar(-heading_vector[2], heading_vector[0])[1], 4)

                    pose = (x, start_pose_y, z, phi)

                    goal_position = episode['goals'][0]['position']

                    geodesic_distance = episode['info']['geodesic_distance']

                    start_goal_pair = (
                        pose, (goal_position[0], goal_position[2]), geodesic_distance)

                    if 'start_goal_pair' in scene_floor_dict[scene_with_index][idx_floor]:
                        scene_floor_dict[scene_with_index][idx_floor]['start_goal_pair'].append(
                            start_goal_pair)
                    else:
                        scene_floor_dict[scene_with_index][idx_floor]['start_goal_pair'] = [
                            start_goal_pair]

                    break

    for idx_floor in list(scene_floor_dict[scene_with_index].keys()):
        assert len(scene_floor_dict[scene_with_index]
                   [idx_floor]['start_goal_pair']) > 0
        # save the first 100 start_goal_pair in case the file is too large
        start_goal_pairs = scene_floor_dict[scene_with_index][idx_floor]['start_goal_pair']
        if len(start_goal_pairs) > 100:
            scene_floor_dict[scene_with_index][idx_floor]['start_goal_pair'] = start_goal_pairs[0:100]


np.save(f'{saved_folder}/{split}_scene_floor_dict.npy', scene_floor_dict)
