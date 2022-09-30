import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor, degrees
import random
from modeling.utils.navigation_utils import change_brightness, SimpleRLEnv, get_obs_and_pose, get_obs_and_pose_by_action
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, pxl_coords_to_pose, gen_arrow_head_marker, read_map_npy, read_occ_map_npy, plus_theta_fn, convertInsSegToSSeg, create_folder
from modeling.utils.map_utils_pcd_height import SemanticMap
from modeling.utils.frontier_utils import prune_skeleton_graph
from modeling.localNavigator_Astar import localNav_Astar
import habitat
import habitat_sim
import random
from core import cfg
from modeling.localNavigator_slam import localNav_slam
from skimage.morphology import skeletonize
import sknw
import networkx as nx
import bz2
import _pickle as cPickle

def display_skeleton(occupancy_grid, graph):
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
	ax.imshow(occupancy_grid, cmap='gray')

	# draw edges by pts
	for (s,e) in graph.edges():
		ps = graph[s][e]['pts']
		ax.plot(ps[:,1], ps[:,0], 'green')
		
	# draw node by o
	nodes = graph.nodes()
	ps = np.array([nodes[i]['o'] for i in nodes])
	ax.plot(ps[:,1], ps[:,0], 'r.')

	# title and show
	plt.title('Build Graph')
	fig.tight_layout()
	plt.show()

def my_tsp(G, weight="weight"):
	method = nx.algorithms.approximation.christofides
	nodes = list(G.nodes)

	dist = {}
	path = {}
	for n, (d, p) in nx.all_pairs_dijkstra(G, weight=weight):
		dist[n] = d
		path[n] = p

	GG = nx.Graph()
	for u in nodes:
		for v in nodes:
			if u == v:
				continue
			GG.add_edge(u, v, weight=dist[u][v])
	best_GG = method(GG, weight)

	best_path = []
	for u, v in nx.utils.pairwise(best_GG):
		best_path.extend(path[u][v][:-1])
	best_path.append(v)
	return best_path

cfg.merge_from_file('configs/exp_specialization.yaml')
cfg.freeze()

scene_name = 'zsNo4HB9uLZ_0'
occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/val/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)

#=========================== get the skeleton image of the whole map ==========================
skeleton = skeletonize(gt_occ_map)
skeleton_G = sknw.build_sknw(skeleton)
skeleton_G = prune_skeleton_graph(skeleton_G)
#display_skeleton(gt_occ_map, skeleton_G)

#======================= get the largest connected component of the graph =================
sub_G = skeleton_G.subgraph(max(nx.connected_components(skeleton_G), key=len))
#display_skeleton(gt_occ_map, sub_G)

#======================= compute tsp of traveling the graph ============================
path = my_tsp(sub_G)
nodes = list(sub_G.nodes)
for i in range(len(path)):
	if not nodes:
		index = i
		break
	if path[i] in nodes:
		nodes.remove(path[i])
#traverse_path = path[:index]
traverse_path = path
print(f'finished computing TSP ...')
#assert 1==2

#=============================== get observations along the path =============================
split = 'val' #'test' #'train'
env_scene = 'zsNo4HB9uLZ' #'17DRP5sb8fy' #'yqstnuAEVhm'
floor_id = 0
scene_name = 'zsNo4HB9uLZ_0' #'17DRP5sb8fy_0' #'yqstnuAEVhm_0'

scene_floor_dict = np.load(f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy', allow_pickle=True).item()

act_dict = {-1: 'Done', 0: 'stop', 1: 'forward', 2: 'left', 3:'right'}

#================================ load habitat env============================================
config = habitat.get_config(config_paths=cfg.GENERAL.DATALOADER_CONFIG_PATH)
config.defrost()
config.SIMULATOR.SCENE = f'{cfg.GENERAL.HABITAT_SCENE_DATA_PATH}/mp3d/{env_scene}/{env_scene}.glb'
config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
config.freeze()

env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
env.reset()

#============================ get scene ins to cat dict
scene = env.semantic_annotations()
ins2cat_dict = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}

#=================================== start original navigation code ========================
np.random.seed(cfg.GENERAL.RANDOM_SEED)
random.seed(cfg.GENERAL.RANDOM_SEED)

if cfg.NAVI.GT_OCC_MAP_TYPE == 'NAV_MESH':
	occ_map_npy = np.load(f'output/semantic_map/{split}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)
H, W = gt_occ_map.shape[:2]

#===================================== load modules ==========================================
LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

LS = localNav_slam(pose_range, coords_range, WH, mark_locs=True, close_small_openings=False, recover_on_collision=True, 
	fix_thrashing=False, point_cnt=2)
LS.reset(gt_occ_map)

semMap_module = SemanticMap(split, scene_name, pose_range, coords_range, WH, ins2cat_dict) # build the observed sem map
traverse_lst = []
act_lst = []
traverse_pose_lst = []

#===================================== setup the start location ===============================#
scene_height = scene_floor_dict[env_scene][floor_id]['y']
agent_start_coords = sub_G.nodes[traverse_path[0]]['pts'][0, ::-1]
agent_start_pose = pxl_coords_to_pose(agent_start_coords, pose_range, coords_range, WH)
start_pose = (agent_start_pose[0], -agent_start_pose[1], 0.2964) 
output_folder = f'output/scene_traversal'
saved_folder = f'{output_folder}/{scene_name}'
create_folder(saved_folder)

agent_pos = np.array([start_pose[0], scene_height, start_pose[1]]) # (6.6, -6.9), (3.6, -4.5)
# check if the start point is navigable
if not env.is_navigable(agent_pos):
	print(f'start pose is not navigable ...')
	assert 1==2

if cfg.NAVI.HFOV == 90:
	obs_list, pose_list = [], []
	heading_angle = start_pose[2]
	obs, pose = get_obs_and_pose(env, agent_pos, heading_angle)
	obs_list.append(obs)
	pose_list.append(pose)


step = 0
subgoal_coords = None
subgoal_pose = None 
MODE_FIND_SUBGOAL = True
idx_path_node = 1

count_sample = 0

while True:
	print(f'step = {step}')

	#=============================== get agent global pose on habitat env ========================#
	pose = pose_list[-1]
	#print(f'agent position = {pose[:2]}, angle = {pose[2]}')
	agent_map_pose = (pose[0], -pose[1], -pose[2])
	#print(f'agent_map_pose, x = {agent_map_pose[0]}, y = {agent_map_pose[1]}, angle = {np.rad2deg(agent_map_pose[2])}')
	traverse_lst.append(agent_map_pose)
	traverse_pose_lst.append(pose)

	# add the observed area
	semMap_module.build_semantic_map(obs_list, pose_list, step=step, saved_folder=saved_folder)

	#============================ save the observations ===============================
	if True:
		assert len(obs_list) == 1
		obs = obs_list[0]
		rgb_img = obs['rgb']
		depth_img = obs['depth'][:,:,0]
		#print(f'depth_img.shape = {depth_img.shape}')
		InsSeg_img = obs["semantic"]
		sseg_img = convertInsSegToSSeg(InsSeg_img, ins2cat_dict)
		eps_data = {}
		eps_data['rgb'] = rgb_img.copy()
		eps_data['depth'] = depth_img.copy()
		eps_data['sseg'] = sseg_img.copy()

		sample_name = str(count_sample).zfill(5)
		with bz2.BZ2File(f'{saved_folder}/{sample_name}.pbz2', 'w') as fp:
			cPickle.dump(
				eps_data,
				fp
			)
		count_sample += 1

	if idx_path_node == len(traverse_path):
		print('traversed the whole path. Done with navigation.')
		#==================================== visualize the path on the map ==============================
		built_semantic_map, observed_area_flag, _ = semMap_module.get_semantic_map()

		color_built_semantic_map = apply_color_to_map(built_semantic_map, flag_small_categories=True)
		#color_built_semantic_map = change_brightness(color_built_semantic_map, observed_area_flag, value=60)

		#=================================== visualize the agent pose as red nodes =======================
		x_coord_lst, z_coord_lst, theta_lst = [], [], []
		for cur_pose in traverse_lst:
			x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]), pose_range, coords_range, WH)
			x_coord_lst.append(x_coord)
			z_coord_lst.append(z_coord)			
			theta_lst.append(cur_pose[2])

		#'''
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
		ax[0].imshow(observed_occupancy_map, cmap='gray')
		marker, scale = gen_arrow_head_marker(theta_lst[-1])
		ax[0].scatter(x_coord_lst[-1], z_coord_lst[-1], marker=marker, s=(30*scale)**2, c='red', zorder=5)
		ax[0].scatter(x_coord_lst, 
			   z_coord_lst, 
			   c=range(len(x_coord_lst)), 
			   cmap='viridis', 
			   s=np.linspace(5, 2, num=len(x_coord_lst))**2, 
			   zorder=3)
		ax[0].scatter(subgoal_coords[0], subgoal_coords[1], marker='o', c='yellow', zorder=6)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		#ax.set_title('improved observed_occ_map + frontiers')

		ax[1].imshow(color_built_semantic_map)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)

		fig.tight_layout()
		plt.title('observed area')
		#plt.show()
		fig.savefig(f'{saved_folder}/final_maps.jpg', bbox_inches='tight')
		plt.close()
		#assert 1==2
		#'''
		break

	if MODE_FIND_SUBGOAL:

		observed_occupancy_map, gt_occupancy_map, observed_area_flag, built_semantic_map = semMap_module.get_observed_occupancy_map(agent_map_pose)
		subgoal_coords = sub_G.nodes[traverse_path[idx_path_node]]['pts'][0, ::-1]

		#============================================= visualize semantic map ===========================================#
		if step % 1000 == 0:
			#==================================== visualize the path on the map ==============================
			built_semantic_map, observed_area_flag, _ = semMap_module.get_semantic_map()

			color_built_semantic_map = apply_color_to_map(built_semantic_map, flag_small_categories=True)
			#color_built_semantic_map = change_brightness(color_built_semantic_map, observed_area_flag, value=60)

			#=================================== visualize the agent pose as red nodes =======================
			x_coord_lst, z_coord_lst, theta_lst = [], [], []
			for cur_pose in traverse_lst:
				x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]), pose_range, coords_range, WH)
				x_coord_lst.append(x_coord)
				z_coord_lst.append(z_coord)			
				theta_lst.append(cur_pose[2])

			#'''
			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
			ax[0].imshow(observed_occupancy_map, cmap='gray')
			marker, scale = gen_arrow_head_marker(theta_lst[-1])
			ax[0].scatter(x_coord_lst[-1], z_coord_lst[-1], marker=marker, s=(30*scale)**2, c='red', zorder=5)
			ax[0].scatter(x_coord_lst, 
				   z_coord_lst, 
				   c=range(len(x_coord_lst)), 
				   cmap='viridis', 
				   s=np.linspace(5, 2, num=len(x_coord_lst))**2, 
				   zorder=3)
			ax[0].scatter(subgoal_coords[0], subgoal_coords[1], marker='o', c='yellow', zorder=6)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			#ax.set_title('improved observed_occ_map + frontiers')

			ax[1].imshow(color_built_semantic_map)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)

			fig.tight_layout()
			plt.title('observed area')
			plt.show()
			#fig.savefig(f'{saved_folder}/step_{step}_semmap.jpg')
			#plt.close()
			#assert 1==2
			#'''
		
		#print(f'subgoal_coords = {subgoal_coords}')

	# ================================ take next action ====================================
	act, act_seq, _, _ = LS.plan_to_reach_frontier(agent_map_pose, subgoal_coords, 
		observed_occupancy_map)

	#print(f'subgoal_coords = {subgoal_coords}')
	#act = LS.next_action()
	print(f'action = {act_dict[act]}')
	
	if act == -1 or act == 0: # finished navigating to the subgoal
		print(f'reached the subgoal')
		MODE_FIND_SUBGOAL = True
		idx_path_node += 1
	else:
		step += 1
		#print(f'next_pose = {next_pose}')
		#agent_pos = np.array([next_pose[0], scene_height, next_pose[1]])
		# output rot is negative of the input angle
		if cfg.NAVI.HFOV == 90:
			obs_list, pose_list = [], []
			obs, pose = get_obs_and_pose_by_action(env, act)
			obs_list.append(obs)
			pose_list.append(pose)

	act_lst.append(act)

eps_data = {}
eps_data['pose'] = traverse_pose_lst
eps_data['action'] = act_lst
with bz2.BZ2File(f'{saved_folder}/traversal_info.pbz2', 'w') as fp:
	cPickle.dump(
		eps_data,
		fp
	)
count_sample += 1


env.close()