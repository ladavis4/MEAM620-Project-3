import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        # 3.0 and 4.2
        self.k_d = 3.0
        self.K_d = np.array([[self.k_d, 0, 0], [0, self.k_d, 0], [0, 0, self.k_d]])
        self.k_p = 4.2
        self.K_p = np.array([[self.k_p, 0, 0], [0, self.k_p, 0], [0, 0, self.k_p]])

        # 500 and 5 worked well
        # 1000 and 10 did too
        self.K_R_gain = 400
        self.K_w_gain = 4


    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        # parse input
        r = state['x']
        r_dot = state['v']
        q = state['q']
        w = state['w']

        r_T = flat_output['x']
        r_dot_T = flat_output['x_dot']
        r_ddot_T = flat_output['x_ddot']
        psi_T = flat_output['yaw']
        psi_dot_T = flat_output['yaw_dot']

        R = Rotation.from_quat(q).as_matrix()

        # calculate F_des from (33), (32), and (31)
        r_ddot_des = r_ddot_T - self.K_d @ (r_dot - r_dot_T) - self.K_p @ (r - r_T)  # 31
        F_des = self.mass * r_ddot_des + np.array([0, 0, self.mass * self.g])  # 32

        # compute u1 from (34)

        b3 = R @ np.array([[0], [0], [1]])
        u1 = b3.T @ F_des

        # determine R_des from (38) and the definitions of b_des
        b_3_des = F_des / np.linalg.norm(F_des)
        a_psi = np.array([np.cos(psi_T), np.sin(psi_T), 0])
        b_2_des = np.cross(b_3_des, a_psi) / np.linalg.norm(np.cross(b_3_des, a_psi))
        R_des = np.array([np.cross(b_2_des, b_3_des), b_2_des, b_3_des])  #This is wrong!!!! 
        R_des = R_des.T


        # find the error orientation vector e_R from (39)
        e_R_mat = .5 * (R_des.T @ R - R.T @ R_des) # *-1 Why do I have a negative 1 here??
        e_R = np.array([e_R_mat[2, 1], e_R_mat[0, 2], e_R_mat[1, 0]]) # v operation

        # compute u2 from (40)
        e_w = w # - w_des is set to zero
        K_R = np.array([[self.K_R_gain, 0, 0], [0, self.K_R_gain, 0], [0, 0, self.K_R_gain]])
        K_w = np.array([[self.K_w_gain, 0, 0], [0, self.K_w_gain, 0], [0, 0, self.K_w_gain]])
        u2 = self.inertia @ ((-1 * K_R @ e_R) - (K_w @ e_w))

        # calculate motor speeds from motor commands
        gamma = self.k_drag/self.k_thrust #m/f
        l = self.arm_length
        A = np.array([[1, 1, 1, 1], [0, l, 0, -l], [-l, 0, l, 0], [gamma, -gamma, gamma, -gamma]])
        B = np.array([[u1[0]], [u2[0]], [u2[1]], [u2[2]]])
        motor_forces = np.linalg.inv(A) @ B
        motor_forces = motor_forces.reshape(-1)

        neg = motor_forces<0

        cmd_motor_speeds = np.sqrt(np.abs(motor_forces) / self.k_thrust)
        cmd_motor_speeds[neg] = cmd_motor_speeds[neg] * -1

        cmd_thrust = u1
        cmd_moment = u2
        cmd_q = Rotation.from_matrix(R_des).as_quat()


        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input

