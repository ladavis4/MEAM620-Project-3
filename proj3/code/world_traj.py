import numpy as np
from .graph_search import graph_search
import scipy

class WorldTraj(object):
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.25

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        #self.points = np.zeros((1,3)) # shape=(n_pts,3)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        points = np.array(rdp(self.path))
        if np.linalg.norm(points[0] - points[2]) <= np.linalg.norm(points[1] - points[2]):
           points = np.delete(points, 1, axis=0)

        if np.linalg.norm(points[-2] - points[-1]) >= np.linalg.norm(points[-3] - points[-1]):
            points = np.delete(points, -2, axis=0)

        #Delete close points
        points = remove_close_points(points, thresh=.35)

        dist = np.linalg.norm(points[1:, :] - points[:-1, :], axis=1)
        vel = 1.6
        self.points = points
        self.num_points = points.shape[0]
        m = points.shape[0] - 1  # number of segments
        # Find the time for each point
        travel_time = dist / vel
        self.t_start = np.cumsum(travel_time)
        self.t_start = np.insert(self.t_start, 0, 0)

        # Scale time to 0-1
        t = travel_time
        t = np.insert(t, 0, 0)
        self.t = t

        b = np.zeros([3, 3])
        A = np.zeros([3, m * 6])

        # Add start point constraints
        A[0:3, 0:6] = np.array([[0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 2, 0, 0]])
        b[0, :] = points[0, :]
        b[1, :] = np.zeros([1, 3])
        b[2, :] = np.zeros([1, 3])

        # Add end point constraints
        mat = np.zeros([3, m * 6])
        mat[:, -6:] = np.array([[t[-1] ** 5, t[-1] ** 4, t[-1] ** 3, t[-1] ** 2, t[-1], 1],
                                [5 * t[-1] ** 4, 4 * t[-1] ** 3, 3 * t[-1] ** 2, 2 * t[-1], 1, 0],
                                [20 * t[-1] ** 3, 12 * t[-1] ** 2, 6 * t[-1], 2, 0, 0]])
        A = np.append(A, mat, axis=0)
        b = np.append(b, np.array([points[-1], np.zeros(3), np.zeros(3)]), axis=0)

        # Add intermediate point constraints
        for i in range(m - 1):
            t_val = i + 1
            mat = np.zeros([2, m * 6])
            mat[0, (i * 6):(i + 1) * 6] = np.array(
                [[t[t_val] ** 5, t[t_val] ** 4, t[t_val] ** 3, t[t_val] ** 2, t[t_val], 1]])
            mat[1, (i + 2) * 6 - 1] = 1
            A = np.append(A, mat, axis=0)
            b = np.append(b, np.array([points[t_val], points[t_val]]), axis=0)

        # Add continuity constraints
        for i in range(m - 1):
            t_val = i + 1
            mat = np.zeros([4, m * 6])
            mat[:, (i * 6):(i + 2) * 6] = np.array(
                [[5 * t[t_val] ** 4, 4 * t[t_val] ** 3, 3 * t[t_val] ** 2, 2 * t[t_val], 1, 0, 0, 0, 0, 0, -1, 0],
                 [20 * t[t_val] ** 3, 12 * t[t_val] ** 2, 6 * t[t_val], 2, 0, 0, 0, 0, 0, -2, 0, 0],
                 [60 * t[t_val] ** 2, 24 * t[t_val], 6, 0, 0, 0, 0, 0, -6, 0, 0, 0],
                 [120 * t[t_val], 24, 0, 0, 0, 0, 0, -24, 0, 0, 0, 0]])

            A = np.append(A, mat, axis=0)
            b = np.append(b, np.array([np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]), axis=0)

        self.c = scipy.linalg.solve(A, b)
        print(self.c)

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        # scale t to 0->1

        # check what segment the quad is in
        if t > self.t_start[-1]:
            x = self.points[-1]
            x_dot = np.zeros((3,))
        else:
            for i in range(self.num_points - 1):
                if self.t_start[i] <= t < self.t_start[i + 1]:
                    # calculate velocity and position
                    t = t - self.t_start[i]
                    T_x = np.array([t ** 5, t ** 4, t ** 3, t ** 2, t, 1])
                    T_xd = np.array([5 * t ** 4, 4 * t ** 3, 3 * t ** 2, 2 * t, 1, 0])
                    T_xdd = np.array([20 * t ** 3, 12 * t ** 2, 6 * t, 2, 0, 0])
                    x = T_x @ self.c[i * 6:(i + 1) * 6, :]
                    x_dot = T_xd @ self.c[i * 6:(i + 1) * 6, :]
                    x_ddot = T_xdd @ self.c[i*6:(i+1)*6, :]
                    break

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output


def rdp(points, epsilon=0.3):
    # get the start and end points
    start = np.tile(np.expand_dims(points[0], axis=0), (points.shape[0], 1))
    end = np.tile(np.expand_dims(points[-1], axis=0), (points.shape[0], 1))

    # find distance from other_points to line formed by start and end
    dist_point_to_line = np.linalg.norm(np.cross(points - start, end - start), axis=-1) / np.linalg.norm(end - start, axis=-1)
    # get the index of the points with the largest distance
    max_idx = np.argmax(dist_point_to_line)
    max_value = dist_point_to_line[max_idx]

    result = []
    if max_value > epsilon:
        partial_results_left = rdp(points[:max_idx+1], epsilon)
        result += [list(i) for i in partial_results_left if list(i) not in result]
        partial_results_right = rdp(points[max_idx:], epsilon)
        result += [list(i) for i in partial_results_right if list(i) not in result]
    else:
        result += [points[0], points[-1]]

    return result

def remove_close_points(points, thresh = 1.0):
    done = False
    while not done:
        num_points = points.shape[0]
        for i in range(num_points - 2):
            dist = np.linalg.norm(points[i + 1] - points[i])
            if dist < thresh:
                points = np.delete(points, i+1, 0)
                break
        if i == num_points-3:
            done = True

    done = False
    while not done:
        if np.linalg.norm(points[-1] - points[-2]) < thresh:
            points = np.delete(points, -2, 0)
        else:
            done = True
    return points