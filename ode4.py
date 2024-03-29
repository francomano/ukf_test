import math
import numpy as np
import pinocchio as pin

def f(model, data, tau, x, iL, iR, wheel_distance=0.2022, wheel_radius=0.0985):
    '''
    Returns the derivative of the state vector x

    Arguments:
        - model: the pinocchio model
        - data: the pinocchio data
        - G: the null space matrix of the differential drive
        - tau: the control input
        - x: the state vector [x, y, theta, w_l, w_r]
        - semi_axle: the semi-axle length of the differential drive
        - wheel_radius: the radius of the wheels
    '''

    # Split the state vector into the joint configuration and the joint velocity
    q = x[:3]
    v = x[3:]
    theta = q[2]

    S = (1 / (2 * wheel_distance)) * np.array([
        [wheel_distance * wheel_radius * (1 - iL) * math.cos(theta), wheel_distance * wheel_radius * (1 - iR) * math.cos(theta)],
        [wheel_distance * wheel_radius * (1 - iL) * math.sin(theta), wheel_distance * wheel_radius * (1 - iR) * math.sin(theta)],
        [-2*wheel_radius * (1 - iL), 2*wheel_radius * (1 - iR)]
    ]) 
    
    B = np.array([
        [math.cos(theta), math.cos(theta)],
        [math.sin(theta), math.sin(theta)],
        [-wheel_distance/2, wheel_distance/2]
    ])
    
    # Extract centroidal mass and centroidal inertia matrix
    m = data.Ig.mass                  # Total mass of the robot
    I_zz = data.Ig.inertia[2, 2]      # Moment of inertia about the vertical axis    

    # Compute the 3x3 mass matrix M
    M = np.array([[m, 0, 0], [0, m, 0], [0, 0, I_zz]])
    
    Bbar = S.T @ B
    Mbar = S.T @ M @ S
    
    # Compute the derivative of the state vector
    dx = np.zeros_like(x)

    dx[:3] = S @ v                                  # [x_dot, y_dot, theta_dot]
    dx[3:] = np.linalg.inv(Mbar) @ Bbar @ tau       # [w_l_dot, w_r_dot]

    return dx 

def ode4(model, data, x0, dt, tau, iL=0, iR=0):
    # compute runge kutta solutions
    s1 = f(model, data, tau, x0, iL, iR)
    x1 = x0 + dt/2*s1

    s2 = f(model, data, tau, x1, iL, iR)
    x2 = x0 + dt/2*s2

    s3 = f(model, data, tau, x2, iL, iR)
    x3 = x0 + dt*s3

    s4 = f(model, data, tau, x3, iL, iR)
    x = x0 + dt*(s1 + 2*s2 + 2*s3 + s4)/6

    return x


def quat_to_rpy(q):  
    '''
    Convert the state vector from quaternion to RPY (roll, pitch, yaw) 
    and wheel cos and sin to wheel angles

    Arguments:
        - q: the state vector [x, y, z, quaternion, cos(left wheel),sin(left wheel), cos(right wheel), sin(right wheel)]
    '''  
    # Convert the quaternion to a rotation matrix
    rotation_matrix = pin.Quaternion(q[6], q[3], q[4], q[5]).toRotationMatrix()
    # Convert the rotation matrix to RPY (roll, pitch, yaw)
    rpy = pin.rpy.matrixToRpy(rotation_matrix)

    # Convert the wheel angles to cos and sin
    left_wheel = math.atan2(q[8], q[7])
    right_wheel = math.atan2(q[10], q[9])

    # Return [x, y, z, roll, pitch, yaw, left wheel_rad, right wheel_rad]
    return np.array([q[0], q[1], q[2], rpy[0], rpy[1], rpy[2], left_wheel, right_wheel])

def rpy_to_quat(q):
    '''
    Convert the state vector from RPY (roll, pitch, yaw) to quaternion
    and wheel angles to cos and sin

    Arguments:
        - q: the state vector [x, y, z, roll, pitch, yaw, left wheel_rad, right wheel_rad]
    '''
    # Convert the RPY (roll, pitch, yaw) to a rotation matrix
    rotation_matrix = pin.rpy.rpyToMatrix(q[3], q[4], q[5])
    # Convert the rotation matrix to a quaternion
    quat = pin.Quaternion(rotation_matrix)

    # Convert the wheel angles to cos and sin
    sin_l = math.sin(q[6])
    cos_l = math.sin(q[6])
    sin_r = math.sin(q[7])
    cos_r = math.sin(q[7])
    
    # [x, y, z, quaternion, cos(left wheel),sin(left wheel), cos(right wheel), sin(right wheel)]
    return np.array([q[0], q[1], q[2], quat.x, quat.y, quat.z, quat.w, cos_l, sin_l, cos_r, sin_r])
    