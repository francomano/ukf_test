import math
import numpy as np
import pinocchio as pin

def f(model, data, G, tau, x, semi_axle=0.2022, wheel_radius=0.0985):
    '''
    Returns the derivative of the state vector x

    Arguments:
        - model: the pinocchio model
        - data: the pinocchio data
        - G: the null space matrix of the differential drive
        - tau: the control input
        - x: the state vector [q, v] with: 
                q = [x, y, z, roll, pitch, yaw, left_wheel, right_wheel]
                v = [v_x, v_y, v_z, w_x, w_y, w_z, w_l, w_r]
        - semi_axle: the semi-axle length of the differential drive
        - wheel_radius: the radius of the wheels
    '''

    # Split the state vector into the joint configuration and the joint velocity
    q = x[:model.nv]
    v = x[model.nv:]

    # Convert the joint configuration to a quaternion (pinocchio uses quaternions)
    q_pin = rpy_to_quat(q)

    
    # Null space matrix of differential drive (x, y, z, roll, pitch, yaw)
    G[0, 0] = G[0, 1] = wheel_radius*np.cos(q[5])/2     # r * cos(theta) / 2
    G[1, 0] = G[1, 1] = wheel_radius*np.sin(q[5])/2     # r * sin(theta) / 2
    G[5, 0] = -wheel_radius/(2*semi_axle)               # -r / (2 * l)
    G[5, 1] = wheel_radius/(2*semi_axle)                # r / (2 * l)
    
    # Compute the derivative of the joint configuration
    q_dot = np.dot(G, v)
    
    # Compute the upper triangular part of the joint space inertia matrix M 
    Mq = pin.crba(model, data, q_pin)

    # 
    data.M[np.tril_indices(model.nv, -1)] = data.M.transpose()[np.tril_indices(model.nv, -1)]
    Mq = data.M

    

    Damp = np.eye(model.nv)
    for i in range(model.nv):
        Damp[i, i] = model.damping[i]
        
    # Recursive Newton-Euler algorithm (inverse dynamics, aka the joint torques)
    nq = pin.rnea(model, data, q_pin, q_dot, np.zeros(model.nv)) + np.dot(Damp, q_dot)

    # print("nq", nq)

    print("G", G)

    # reduced model
    M = np.dot(np.dot(G.transpose(), Mq), G)
    #n = np.dot(S.transpose(), np.dot(Mq, np.dot(dS, nu)) + nq)
    n = np.dot(G.transpose(), nq)   # omit np.dot(Mq, np.dot(dS, nu))

    # PROVVISORIO
    tau_new = np.array([0, 0, 0, 0, 0, 0, tau[0], tau[1]])

    print("M", M)

    # Compute the derivative of the state vector
    dx = np.zeros_like(x)
    dx[:model.nv] = q_dot
    dx[model.nv:] = np.linalg.pinv(M).dot(tau_new - n)

    print("dx", dx)
    

    return dx

def ode4(model, data, x0, dt, tau, G):
    # compute runge kutta solutions
    s1 = f(model, data, G, tau, x0)
    x1 = x0 + dt/2*s1

    s2 = f(model, data, G, tau, x1)
    x2 = x0 + dt/2*s2

    s3 = f(model, data, G, tau, x2)
    x3 = x0 + dt*s3

    s4 = f(model, data, G, tau, x3)
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
    