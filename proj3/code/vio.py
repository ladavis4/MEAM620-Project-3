#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    new_p = p + v * dt + .5 * (q.as_matrix() @ (a_m - a_b) + g) * dt ** 2
    new_v = v + (q.as_matrix() @ (a_m - a_b) + g) * dt
    new_q = q * Rotation.from_rotvec((w_m - w_b).flatten() * dt)

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    R = q.as_matrix()

    # Construct F_x matrix
    F_x = np.zeros((18, 18))
    F_x[0:3, 0:3] = np.eye(3)
    F_x[0:3, 3:6] = np.eye(3) * dt
    F_x[3:6, 3:6] = np.eye(3)
    F_x[3:6, 6:9] = -R @ vec2skew(a_m - a_b) * dt
    F_x[3:6, 9:12] = -R * dt
    F_x[3:6, 15:18] = np.eye(3) * dt
    F_x[6:9, 6:9] = Rotation.from_rotvec(((w_m - w_b) * dt).flatten()).as_matrix().T
    F_x[6:9, 12:15] = -np.eye(3) * dt
    F_x[9:, 9:] = np.eye(9)

    # Construct F_i matrix
    F_i = np.zeros((18, 12))
    F_i[3:15, :] = np.eye(12)

    # Construct Q matrix
    Q_i = np.zeros((12, 12))
    Q_i[0:3, 0:3] = np.eye(3) * accelerometer_noise_density ** 2 * dt ** 2
    Q_i[3:6, 3:6] = np.eye(3) * gyroscope_noise_density ** 2 * dt ** 2
    Q_i[6:9, 6:9] = np.eye(3) * accelerometer_random_walk ** 2 * dt
    Q_i[9:12, 9:12] = np.eye(3) * gyroscope_random_walk ** 2 * dt

    # return an 18x18 covariance matrix
    return F_x @ error_state_covariance @ F_x.T + F_i @ Q_i @ F_i.T


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    innovation = np.zeros((2, 1))

    # Compute the innovation
    P_c = q.as_matrix().T @ (Pw - p)
    innovation[0] = uv[0] - P_c[0] / P_c[2]
    innovation[1] = uv[1] - P_c[1] / P_c[2]

    if np.linalg.norm(innovation) < error_threshold:
        # Construct H matrix
        dz_dP = 1/P_c[2][0] * np.array([[1, 0, -P_c[0][0]/P_c[2][0]], [0, 1, -P_c[1][0]/P_c[2][0]]])
        dP_ddp = -q.as_matrix().T
        dP_ddtheta = vec2skew(q.as_matrix().T @ (Pw - p).flatten())

        H_t = np.zeros((2, 18))
        H_t[:, 0:3] = dz_dP @ dP_ddp
        H_t[:, 6:9] = dz_dP @ dP_ddtheta

        K_t = error_state_covariance @ H_t.T @ np.linalg.inv(H_t @ error_state_covariance @ H_t.T + Q)
        delta_x = K_t @ innovation
        p = p + delta_x[0:3]
        v = v + delta_x[3:6]
        q = q * Rotation.from_rotvec(delta_x[6:9].flatten())
        a_b = a_b + delta_x[9:12]
        w_b = w_b + delta_x[12:15]
        g = g + delta_x[15:18]

        error_state_covariance = (np.eye(18) - K_t @ H_t) @ error_state_covariance @ (np.eye(18) - K_t @ H_t).T + K_t @ Q @ K_t.T

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

def vec2skew(v):
    """
    Function to convert a 3x1 vector into a 3x3 skew symmetric matrix

    :param v: 3x1 vector
    :return: 3x3 skew symmetric matrix
    """
    v = v.flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
