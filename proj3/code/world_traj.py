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

        # Declare inputs
        debug = False
        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.30
        min_vel = 1.4
        max_vel = 2.0

        # SPLINE PARAMETERS
        epsilon_val = .9
        collision_threshold = .30
        new_point_mode = 0 # 0 = add midpoint, 1 = add point close to collision


        ## USE ASTAR AND RDP TO RETURN POINTS ##
        # Return dense path
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        astar_path = add_extra_points(self.path)

        # Use RDP to prune points
        points = np.array(rdp(self.path, epsilon=epsilon_val))

        # Remove the  closest point to the start/end point if the 2nd closest is closer
        if np.linalg.norm(points[0] - points[2]) <= np.linalg.norm(points[1] - points[2]):
           points = np.delete(points, 1, axis=0)
        if np.linalg.norm(points[-2] - points[-1]) >= np.linalg.norm(points[-3] - points[-1]):
            points = np.delete(points, -2, axis=0)

        # Delete points that are within a threshold of eachother
        points = remove_close_points(points, thresh=.5)

        ## SOLVE FOR TRAJECTORY ##
        self.points = points
        dist = np.linalg.norm(self.points[1:, :] - self.points[:-1, :], axis=1) # distance of segments
        self.num_points = self.points.shape[0]
        m = self.points.shape[0] - 1  # number of segments

        # Find the time for each point
        travel_time = dist / min_vel  # travel time of each segment
        self.t_start = np.cumsum(travel_time)  # time to arrive at each point
        self.t_start = np.insert(self.t_start, 0, 0)

        # Solve for trajectory
        travel_time = np.insert(travel_time, 0, 0)
        self.c = solve_for_trajectory(self.points, travel_time, m)

        # Check if trajectory collides with walls
        num_samples = 100
        x_test = np.zeros((num_samples, 3))
        t_test = np.linspace(0, self.t_start[-1], num=num_samples)
        for i in range(num_samples):
            time = t_test[i]
            out = self.update(time)
            x_test[i, :] = out['x']

        collisions = world.path_collisions(x_test, collision_threshold)
        collision = collisions.size != 0

        if debug:
            from matplotlib.lines import Line2D
            import matplotlib.pyplot as plt
            from flightsim.axes3ds import Axes3Ds

            # Visualize the original dense path from A*, your sparse waypoints, and the
            # smooth trajectory.
            fig = plt.figure('A* Path, Waypoints, and Trajectory')
            ax = Axes3Ds(fig)
            world.draw(ax)
            ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
            ax.plot([goal[0]], [goal[1]], [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
            world.draw_line(ax, self.path, color='red', linewidth=1)
            world.draw_points(ax, self.points, color='purple', markersize=8)
            world.draw_line(ax, x_test, color='black', linewidth=2)
            ax.legend(handles=[
                Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
                Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='Sparse Waypoints'),
                Line2D([], [], color='black', linewidth=2, label='Trajectory')],
                loc='upper right')
            plt.show()

        while collision:
            # find which two points its closest to
            collision_point = collisions[0]
            print(f"Collision detected at {collision_point}")
            t_collision = t_test[np.argmin(np.linalg.norm(x_test - collision_point, axis=1))]
            pt_idx_before = np.where(t_collision - self.t_start > 0, t_collision - self.t_start, np.inf).argmin()
            pt_idx_after = pt_idx_before + 1

            # Get candidate points
            astar_before = np.where(np.all(self.points[pt_idx_before] == astar_path, axis=1))[0][0]
            astar_after = np.where(np.all(self.points[pt_idx_after] == astar_path, axis=1))[0][0]
            candidate_pts = astar_path[astar_before:(astar_after+1)]

            # Get candidate closest to midpoint
            if new_point_mode == 0:
                new_point = get_new_midpoint(candidate_pts, self.points[pt_idx_before], self.points[pt_idx_after])
            elif new_point_mode == 1:
                new_point = get_new_collision(candidate_pts, collision_point)


            #Add new point to points
            self.points = np.insert(self.points, pt_idx_after, new_point, axis=0)

            dist = np.linalg.norm(self.points[1:, :] - self.points[:-1, :], axis=1)  # distance of segments
            self.num_points = self.points.shape[0]
            m = self.points.shape[0] - 1  # number of segments

            # Calculate the travel time for each point
            dist_diff = np.max(dist) - np.min(dist)
            scaler = (max_vel - min_vel) / dist_diff

            vel = min_vel + scaler * (dist - np.min(dist))
            travel_time = dist / vel
            self.t_start = np.cumsum(travel_time)  # time to arrive at each point
            self.t_start = np.insert(self.t_start, 0, 0)

            # Solve for trajectory
            travel_time = np.insert(travel_time, 0, 0)
            self.c = solve_for_trajectory(self.points, travel_time, m)


            #Check collision
            # Check if trajectory collides with walls
            num_samples = 1000
            x_test = np.zeros((num_samples, 3))
            t_test = np.linspace(0, self.t_start[-1], num=num_samples)
            for i in range(num_samples):
                time = t_test[i]
                out = self.update(time)
                x_test[i, :] = out['x']

            collisions = world.path_collisions(x_test, collision_threshold)
            collision = collisions.size != 0

            if debug:
                from matplotlib.lines import Line2D
                import matplotlib.pyplot as plt
                from flightsim.axes3ds import Axes3Ds

                # Visualize the original dense path from A*, your sparse waypoints, and the
                # smooth trajectory.
                fig = plt.figure('A* Path, Waypoints, and Trajectory')
                ax = Axes3Ds(fig)
                world.draw(ax)
                ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3,
                        markerfacecolor='none')
                ax.plot([goal[0]], [goal[1]], [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
                world.draw_line(ax, self.path, color='red', linewidth=1)
                world.draw_points(ax, self.points, color='purple', markersize=8)
                world.draw_line(ax, x_test, color='black', linewidth=2)
                ax.legend(handles=[
                    Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
                    Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='Sparse Waypoints'),
                    Line2D([], [], color='black', linewidth=2, label='Trajectory')],
                    loc='upper right')
                plt.show()


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
        # check what segment the quad is in
        if t >= self.t_start[-1]:
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
                    T_xddd = np.array([60 * t ** 2, 24 * t, 6, 0, 0, 0])

                    x = T_x @ self.c[i * 6:(i + 1) * 6, :]
                    x_dot = T_xd @ self.c[i * 6:(i + 1) * 6, :]
                    x_ddot = T_xdd @ self.c[i*6:(i+1)*6, :]
                    x_dddot = T_xddd @ self.c[i*6:(i+1)*6, :]
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

def solve_for_trajectory(points, t, m):
    """
    INPUTS:
        points - sparese A* points
        t - time for each segment
        m - number of segments
    OUTPUTS:
        c - constraints for the trajectory
    """
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

    c = scipy.linalg.solve(A, b)
    return c


def add_extra_points(points, extra_pts_per_segment=10):
    """
    Adds extra points to the contour. This is done by using np linspace between each point
    :param points: nx2 array of contour points
    :return: nx2 array of interpolated  points
    """
    points_new = points.copy()

    for i in range(points.shape[0] - 1):
        points_add = np.linspace(points[i, :], points[i+1, :], extra_pts_per_segment)
        points_add = points_add[1:-1, :]

        points_new = np.insert(points_new, i + 1 + (i * (extra_pts_per_segment-2)), points_add, axis=0)

    return points_new

def get_new_midpoint(candidate_points, point_before, point_after):
    """
    Returns a new point for the spline at the midpoint between the two existing points on the spline

    :return: new point
    """
    midpoint = np.mean([point_before, point_after], axis=0)
    candidiate_idx = np.argmin(np.linalg.norm(midpoint - candidate_points, axis=1))
    new_point = candidate_points[candidiate_idx]

    return new_point

def get_new_collision(candidate_points, collision_point):
    """
   Returns a new point for the spline closest to the collision point of the drone

   :return: new point
   """

    # Get candidate closest to collision point
    candidiate_idx = np.argmin(np.linalg.norm(collision_point - candidate_points, axis=1))
    new_point = candidate_points[candidiate_idx]

    return new_point