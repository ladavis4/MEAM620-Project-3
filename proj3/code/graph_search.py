from heapq import heappush, heappop  # Recommended.
import numpy as np
import heapq

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    ### INPUTS ###
    norm = 2
    height_val = 6 #m
    height_int = int(height_val/resolution[2])
    height_percentage = .7

    returned_path = False
    while not returned_path:
        occ_map = OccupancyMap(world, resolution, margin)
        occ_map.map = add_upper_bound(occ_map.map, height_int, height_percentage)
        path, nodes_expanded = return_path_and_nodes(occ_map, start, goal, resolution, astar, norm)
        if path is not None:
            returned_path = True
            print(f"Path found")
        else:
            height_percentage -= 0.1
            print(f"No path found, decreasing height_percentage to: {height_percentage}")

    return path, nodes_expanded


def heur(current, target, resolution, norm):
    current_position = current * resolution
    target_position = target * resolution

    if norm == 1:
        return np.linalg.norm(current_position - target_position, ord=1)
    elif norm == 2:
        return np.linalg.norm(current_position - target_position)


def cost(current_position, target_position, resolution):
    current_position = np.array(current_position)
    target_position = np.array(target_position)
    return np.linalg.norm((target_position - current_position) * resolution)


def construct_neighbor_matrix():
    mat = np.array([[0, 0, 0]])
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i == 0 and j == 0 and k == 0:
                    continue
                mat = np.append(mat, [[i, j, k]], axis=0)

    mat = np.delete(mat, 0, 0)
    return mat


def return_neighbors(index, mshape, neighbor_mat, map):
    # returns flat index neighbors
    neighbors = index + neighbor_mat

    neighbors = neighbors[np.where(~np.any(neighbors < 0, axis=1))]

    neighbors = neighbors[np.where(np.logical_and(neighbors[:, 0] < mshape[0], neighbors[:, 0] >= 0))]
    neighbors = neighbors[np.where(np.logical_and(neighbors[:, 1] < mshape[1], neighbors[:, 1] >= 0))]
    neighbors = neighbors[np.where(np.logical_and(neighbors[:, 2] < mshape[2], neighbors[:, 2] >= 0))]

    neighbors = neighbors[~map[tuple(neighbors.T)]]

    return neighbors.tolist()

def add_upper_bound(occ_map, threshold=30, height_percentage = .7):
    """
    This function creates a lower obstacle on the occumap when there is a distance greater than the threshold to the nearest north obstacle
    INPUTS:
        occ_map
        threshold
    OUTPUTS
        occ_map_new
    """

    height_percentage = 1 - height_percentage
    height_threshold = int(occ_map.shape[2] * height_percentage)

    occ_map_new = occ_map.copy()
    top_view = np.any(occ_map, axis=2)

    for j in range(occ_map.shape[0]):
        row = top_view[j, :]
        nearest_north_wall = np.zeros_like(row, dtype=int)
        wall_idxs = np.where(row)[0]
        wall_idxs= np.append(wall_idxs, len(row+1))
        wall_cnt = 0
        for i in range(len(row)):
            nearest_north_wall[i] = wall_idxs[wall_cnt]
            if i >= wall_idxs[wall_cnt]:
                wall_cnt += 1
        dist = nearest_north_wall - np.arange(len(row))
        fix_idx = dist > threshold

        occ_map_new[j, fix_idx, height_threshold:] = True
    return occ_map_new

def return_path_and_nodes(occ_map, start, goal, resolution, astar, norm):
    m_shape = occ_map.map.shape
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    construct_neighbor_matrix()

    Q = [] #this is the heapq
    g = np.zeros(m_shape)
    f = np.zeros_like(g)
    p = np.full((m_shape[0], m_shape[1], m_shape[2], 3), np.nan)
    g[:] = float('inf')
    f[:] = float('inf')
    g[start_index] = 0

    heapq.heappush(Q, (g[start_index], start_index))
    nodes_expanded = 0

    neighbor_mat = construct_neighbor_matrix()

    while Q:
        cur_metric, u = heapq.heappop(Q)

        if astar:
            if cur_metric > f[tuple(u)]:
                continue
        else:
            if cur_metric > g[tuple(u)]:
                continue

        if tuple(u) == goal_index:
            break

        neighbors = return_neighbors(u, m_shape, neighbor_mat, occ_map.map)
        nodes_expanded += 1
        for v in neighbors:
            d = g[tuple(u)] + cost(u, v, resolution)
            if d < g[tuple(v)]:
                g[tuple(v)] = d
                p[tuple(v)] = u

                if astar:
                    h = heur(v, goal_index, resolution, norm)
                    f_val = d + h
                    f[tuple(v)] = f_val
                    heapq.heappush(Q, (f_val, v))
                else:
                    heapq.heappush(Q, (d, v))


    # reconstruct path
    path_solved = True

    cur_val = goal_index
    traj_idx = [cur_val]
    while not cur_val == start_index:
        idx = p[cur_val].astype(int)
        if np.any(idx < 0):
            path_solved = False
            break
        cur_val = tuple(idx)
        traj_idx.insert(0, cur_val)

    if path_solved:
        out = occ_map.index_to_metric_center(traj_idx)
    else:
        out = None

    if np.any(out):
        if not np.array_equal(out[0, :], start):
            out = np.insert(out, 0, start, axis=0)
        if not np.array_equal(out[-1, :], goal) :
            out = np.append(out, [goal], axis=0)
    # Return a tuple (path, nodes_expanded)

    return out, nodes_expanded
