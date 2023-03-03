import numpy as np
import matplotlib.pyplot as plt
from core import cfg
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from core import cfg
import scipy.ndimage
from .baseline_utils import pose_to_coords, apply_color_to_map
from math import sqrt
from operator import itemgetter
import torch
import cv2
from skimage.morphology import skeletonize
import sknw
import networkx as nx
from skimage.graph import MCP_Geometric as MCPG
from skimage.graph import route_through_array
import torch.nn.functional as F
import math


def skeletonize_map(occupancy_grid):
    skeleton = skeletonize(occupancy_grid)

    graph = sknw.build_sknw(skeleton)

    tsp = nx.algorithms.approximation.traveling_salesman_problem
    path = tsp(graph)

    nodes = list(graph.nodes)
    for i in range(len(path)):
        if not nodes:
            index = i
            break
        if path[i] in nodes:
            nodes.remove(path[i])

    d_in = path[:index]
    d_out = path[index-1:]

    cost_din = 0
    for i in range(len(d_in)-1):
        cost_din += graph[d_in[i]][d_in[i+1]]['weight']

    cost_dout = 0
    for i in range(len(d_out)-1):
        cost_dout += graph[d_out[i]][d_out[i+1]]['weight']

    return cost_din+cost_dout, cost_din, cost_dout, skeleton, graph


def skeletonize_frontier_graph(component_occ_grid, skeleton):
    component_skeleton = np.where(component_occ_grid, skeleton, False)

    if np.sum(component_skeleton) > 0:
        component_G = sknw.build_sknw(component_skeleton)

        # ================= computed connected components =============================
        list_ccs = [component_G.subgraph(c).copy()
                    for c in nx.connected_components(component_G)]
        #print(f'len(list_ccs) = {len(list_ccs)}')

        '''	
		plt.imshow(component_occ_grid, cmap='gray')
		for sub_G in set_ccs:
			nodes = sub_G.nodes()
			ps = np.array(nodes)
			plt.plot(ps[:,1], ps[:,0], c=np.random.rand(3,))
		plt.show()
		'''

        # ====================== compute the cost of each component and then add them up
        arr_cost_dall = np.zeros(len(list_ccs))
        arr_cost_din = np.zeros(len(list_ccs))
        arr_cost_dout = np.zeros(len(list_ccs))
        for idx, sub_G in enumerate(list_ccs):
            #print(f'sub_G has {len(sub_G.nodes)} nodes.')
            if len(sub_G.nodes) > 1:  # sub_G has more than one nodes
                path = my_tsp(sub_G)
                # =================== split path into d_in and d_out
                nodes = list(sub_G.nodes)
                for i in range(len(path)):
                    if not nodes:
                        index = i
                        break
                    if path[i] in nodes:
                        nodes.remove(path[i])
                # ================== compute cost_din and cost_dout
                d_in = path[:index]
                d_out = path[index-1:]
                cost_din = 0
                for i in range(len(d_in)-1):
                    cost_din += sub_G[d_in[i]][d_in[i+1]]['weight']
                cost_dout = 0
                for i in range(len(d_out)-1):
                    cost_dout += sub_G[d_out[i]][d_out[i+1]]['weight']
                cost_dall = cost_din + cost_dout
            else:
                cost_din = 1
                cost_dout = 1
                cost_dall = cost_din + cost_dout

            arr_cost_dall[idx] = cost_dall
            arr_cost_din[idx] = cost_din
            arr_cost_dout[idx] = cost_dout

        cost_dall = np.sum(arr_cost_dall)
        cost_din = np.sum(arr_cost_din)
        cost_dout = np.sum(arr_cost_dout)
    else:
        cost_din = 1
        cost_dout = 1
        cost_dall = cost_din + cost_dout
        component_G = nx.Graph()

    return cost_dall, cost_din, cost_dout, component_G


def skeletonize_frontier(component_occ_grid, skeleton):
    skeleton_component = np.where(component_occ_grid, skeleton, False)

    '''
	cp_component_occ_grid = component_occ_grid.copy().astype('int16')
	cp_component_occ_grid[skeleton_component] = 3	
	plt.imshow(cp_component_occ_grid)
	
	plt.show()
	'''

    cost_din = max(np.sum(skeleton_component), 1)
    cost_dout = max(np.sum(skeleton_component), 1)
    cost_dall = (cost_din + cost_dout)

    return cost_dall, cost_din, cost_dout, skeleton_component


def create_dense_graph(skeleton, flag_eight_neighs=True):
    H, W = skeleton.shape
    G = nx.grid_2d_graph(H, W)

    if flag_eight_neighs:
        for edge in G.edges:
            G.edges[edge]['weight'] = 1
        G.add_edges_from([((x, y), (x + 1, y + 1)) for x in range(0, H - 1)
                          for y in range(0, W - 1)] + [((x + 1, y), (x, y + 1))
                                                       for x in range(0, H - 1)
                                                       for y in range(0, W - 1)], weight=1.4)
    # remove those nodes where map is occupied
    mask_occupied_node = (skeleton.ravel() == False)
    nodes_npy = np.array(sorted(G.nodes))
    nodes_occupied = nodes_npy[mask_occupied_node]
    lst_nodes_occupied = list(map(tuple, nodes_occupied))
    G.remove_nodes_from(lst_nodes_occupied)

    return G


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


def skeleton_G_to_skeleton(occ_grid, skeleton_G):
    skeleton = np.zeros(occ_grid.shape, dtype=bool)
    for edge in skeleton_G.edges():
        pts = np.array(skeleton_G.edges[edge]['pts'])
        skeleton[pts[:, 0], pts[:, 1]] = True
    return skeleton


def prune_skeleton(occ_grid, skeleton):
    skeleton_G = sknw.build_sknw(skeleton)
    pruned_skeleton_G = prune_skeleton_graph(skeleton_G)
    skeleton = skeleton_G_to_skeleton(occ_grid, pruned_skeleton_G)
    return skeleton


class Frontier(object):

    def __init__(self, points):
        """Initialized with a 2xN numpy array of points (the grid cell
        coordinates of all points on frontier boundary)."""
        inds = np.lexsort((points[0, :], points[1, :]))
        sorted_points = points[:, inds]
        self.props_set = False
        self.is_from_last_chosen = False
        self.is_obstructed = False
        self.prob_feasible = 1.0
        self.delta_success_cost = 0.0
        self.exploration_cost = 0.0
        self.negative_weighting = 0.0
        self.positive_weighting = 0.0

        self.counter = 0
        self.last_observed_pose = None

        # Any duplicate points should be eliminated (would interfere with
        # equality checking).
        dupes = []
        for ii in range(1, sorted_points.shape[1]):
            if (sorted_points[:, ii - 1] == sorted_points[:, ii]).all():
                dupes += [ii]
        self.points = np.delete(sorted_points, dupes, axis=1)

        # Compute and cache the hash
        self.hash = hash(self.points.tobytes())

        self.R = 1
        self.D = 1.
        self.Din = 1.
        self.Dout = 1.

    def set_props(self,
                  prob_feasible,
                  is_obstructed=False,
                  delta_success_cost=0,
                  exploration_cost=0,
                  positive_weighting=0,
                  negative_weighting=0,
                  counter=0,
                  last_observed_pose=None,
                  did_set=True):
        self.props_set = did_set
        self.just_set = did_set
        self.prob_feasible = prob_feasible
        self.is_obstructed = is_obstructed
        self.delta_success_cost = delta_success_cost
        self.exploration_cost = exploration_cost
        self.positive_weighting = positive_weighting
        self.negative_weighting = negative_weighting
        self.counter = counter
        self.last_observed_pose = last_observed_pose

    @property
    def centroid(self):
        # return self.get_centroid()
        return self.get_frontier_point()

    # '''
    def get_centroid(self):
        """Returns the point that is the centroid of the frontier"""
        centroid = np.mean(self.points, axis=1)
        return centroid

    # '''
    '''
	def get_centroid(self):
		#print(f'points.shape = {self.points.shape}')
		points = self.points.transpose()
		distMatrix = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1)
		centroid_idx = np.argmin(distMatrix.sum(axis=0))
		centroid = self.points[:, centroid_idx]
		return centroid
	'''

    def get_frontier_point(self):
        """Returns the point that is on the frontier that is closest to the
        actual centroid"""
        center_point = np.mean(self.points, axis=1)
        norm = np.linalg.norm(self.points - center_point[:, None], axis=0)
        ind = np.argmin(norm)
        return self.points[:, ind]

    def get_distance_to_point(self, point):
        norm = np.linalg.norm(self.points - point[:, None], axis=0)
        return norm.min()

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return hash(self) == hash(other)


def mask_grid_with_frontiers(occupancy_grid, frontiers, do_not_mask=None):
    """Mask grid cells in the provided occupancy_grid with the frontier points
    contained with the set of 'frontiers'. If 'do_not_mask' is provided, and
    set to either a single frontier or a set of frontiers, those frontiers are
    not masked."""

    if do_not_mask is not None:
        # Ensure that 'do_not_mask' is a set
        if isinstance(do_not_mask, Frontier):
            do_not_mask = set([do_not_mask])
        elif not isinstance(do_not_mask, set):
            raise TypeError("do_not_mask must be either a set or a Frontier")
        masking_frontiers = frontiers - do_not_mask
    else:
        masking_frontiers = frontiers

    masked_grid = occupancy_grid.copy()
    for frontier in masking_frontiers:
        masked_grid[frontier.points[0, :], frontier.points[1, :]] = 2

    return masked_grid


def get_frontiers(occupancy_grid):
    """ detect frontiers from occupancy_grid. 
    """

    filtered_grid = scipy.ndimage.maximum_filter(
        occupancy_grid == cfg.FE.UNOBSERVED_VAL, size=3)
    frontier_point_mask = np.logical_and(filtered_grid,
                                         occupancy_grid == cfg.FE.FREE_VAL)

    if cfg.FE.GROUP_INFLATION_RADIUS < 1:
        inflated_frontier_mask = frontier_point_mask
    else:
        inflated_frontier_mask = gridmap.utils.inflate_grid(
            frontier_point_mask,
            inflation_radius=cfg.FE.GROUP_INFLATION_RADIUS,
            obstacle_threshold=0.5,
            collision_val=1.0) > 0.5

    # Group the frontier points into connected components
    labels, nb = scipy.ndimage.label(
        inflated_frontier_mask, structure=np.ones((3, 3)))

    # Extract the frontiers
    frontiers = set()
    for ii in range(nb):
        raw_frontier_indices = np.where(
            np.logical_and(labels == (ii + 1), frontier_point_mask))
        frontiers.add(
            Frontier(
                np.concatenate((raw_frontier_indices[0][None, :],
                                raw_frontier_indices[1][None, :]),
                               axis=0)))

    return frontiers


def get_frontier_with_maximum_area(frontiers, gt_occupancy_grid):
    """ select frontier with the maximum area from frontiers.

    used for the 'Greedy' strategy.
    """
    if cfg.NAVI.PERCEPTION == 'Potential' or cfg.NAVI.PERCEPTION == 'UNet_Potential':
        max_area = 0
        max_fron = None
        for fron in frontiers:
            if max_fron is None:
                max_area = fron.R
                max_fron = fron
            elif fron.R > max_area:
                max_area = fron.R
                max_fron = fron
            elif fron.R == max_area and hash(fron) > hash(max_fron):
                max_area = fron.R
                max_fron = fron
    return max_fron


def get_the_nearest_frontier(frontiers, agent_pose, dist_occupancy_map, LN):
    """ select nearest frontier to the robot.
    used for the 'FME' strategy.
    """
    agent_coord = LN.get_agent_coords(agent_pose)
    min_L = 10000000
    min_frontier = None

    for fron in frontiers:
        _, L = route_through_array(dist_occupancy_map, (agent_coord[1], agent_coord[0]),
                                   (int(fron.centroid[0]), int(fron.centroid[1])))

        if L < min_L:
            min_L = L
            min_frontier = fron
        elif L == min_L and hash(fron) > hash(min_frontier):
            min_L = L
            min_frontier = fron
    return min_frontier


def count_free_space_at_frontiers(frontiers, gt_occupancy_grid, area=10):
    """ compute the free space in the neighborhoadd of the frontier center.
    """
    H, W = gt_occupancy_grid.shape
    for fron in frontiers:
        centroid = (int(fron.centroid[1]), int(fron.centroid[0]))
        x1 = max(0, centroid[0] - area)
        x2 = min(W, centroid[0] + area)
        y1 = max(0, centroid[1] - area)
        y2 = min(H, centroid[1] + area)
        fron_neigh = gt_occupancy_grid[y1:y2, x1:x2]
        #print(f'centroid[0] = {centroid[0]}, y1 = {y1}, y2= {y2}, x1 = {x1}, x2 = {x2}')
        # plt.imshow(fron_neigh)
        # plt.show()
        fron.area_neigh = np.sum(fron_neigh == cfg.FE.FREE_VAL)
        #print(f'fron.area_neigh = {fron.area_neigh}')


def select_top_frontiers(frontiers, top_n=5):
    """ select a few frontiers with the largest value.

    The objective is to reduce the number of frontiers when using the 'DP' strategy.
    top_n decides the number of frontiers to keep.
    """
    if len(frontiers) <= top_n:
        return frontiers

    lst_frontiers = []
    for fron in frontiers:
        lst_frontiers.append((fron, fron.R))

    res = sorted(lst_frontiers, key=itemgetter(1), reverse=True)[:top_n]

    new_frontiers = set()
    for fron, _ in res:
        new_frontiers.add(fron)

    return new_frontiers


def update_frontier_set_data_gen(old_set, new_set, max_dist=6, chosen_frontier=None):
    for frontier in old_set:
        frontier.is_from_last_chosen = False

    # shallow copy of the set
    old_set = old_set.copy()

    # These are the frontiers that will not appear in the new set
    outgoing_frontier_set = old_set - new_set
    # These will appear in the new set
    added_frontier_set = new_set - old_set

    '''
    # This block of code is useless.
    if max_dist is not None:
        # loop through the newly added frontier set and set properties based upon the outgoing frontier set
        for af in added_frontier_set:
            nearest_frontier, nearest_frontier_dist = _get_nearest_feasible_frontier(
                af, outgoing_frontier_set)
            #print(f'nearest_frontier_dist = {nearest_frontier_dist}')
            if nearest_frontier_dist < max_dist:
                if nearest_frontier == chosen_frontier:
                    af.is_from_last_chosen = True

    if len(added_frontier_set) == 0:
        print(f'*** corner case, no new frontier.')
        chosen_frontier.is_from_last_chosen = True
    '''

    # Remove frontier_set that don't appear in the new set
    old_set.difference_update(outgoing_frontier_set)

    # Add the new frontier_set
    old_set.update(added_frontier_set)

    return old_set, added_frontier_set


def _eucl_dist(p1, p2):
    """Helper to compute Euclidean distance."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def _get_nearest_feasible_frontier(frontier, reference_frontier_set):
    """Returns the nearest 'feasible' frontier from a reference set."""
    f_gen = [(of, _eucl_dist(of.get_centroid(), frontier.get_centroid()))
             for of in reference_frontier_set]
    if len(f_gen) == 0:
        return None, 1e10
    else:
        return min(f_gen, key=lambda fd: fd[1])
