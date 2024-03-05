import math
import numpy as np
import pinocchio as pin

def f(model, data, S, dS, tau, x):
    q = x[:model.nv]
    q_pin = rpy_to_quat(q)
    nu = x[model.nv:]

    semi_axle = 0.2022
    wheel_radius = 0.0985
    theta = q[5]

    # null space Matrix
    S[0, 0] = S[0, 1] = wheel_radius*np.cos(theta)/2
    S[1, 0] = S[1, 1] = wheel_radius*np.sin(theta)/2
    S[5, 0] = -wheel_radius/(2*semi_axle)
    S[5, 1] = wheel_radius/(2*semi_axle)
    S[-nu.size:, :] = np.eye(nu.size)
    
    dq = np.dot(S, nu)
    dtheta = dq[5]

    print("DEQQUUE", dq)

    # derivative of null space matrix
    dS[0, 0] = -dtheta*wheel_radius*np.sin(theta)/2
    dS[0, 1] = -dtheta*wheel_radius*np.sin(theta)/2
    dS[1, 0] = dtheta*wheel_radius*np.cos(theta)/2
    dS[1, 1] = dtheta*wheel_radius*np.cos(theta)/2

    print("Q PINNE", q_pin)

    Mq = pin.crba(model, data, q_pin)
    data.M[np.tril_indices(model.nv, -1)] = data.M.transpose()[np.tril_indices(model.nv, -1)]
    Mq = data.M
    Damp = np.eye(model.nv)
    for i in range(model.nv):
        Damp[i, i] = model.damping[i]

    print("DAMP", Damp)
    print("ALLHOA", np.dot(Damp, dq))
    nq = pin.rnea(model, data, q_pin, dq, np.zeros(model.nv)) + np.dot(Damp, dq)

    # reduced model
    M = np.dot(np.dot(S.transpose(), Mq), S)
    n = np.dot(S.transpose(), np.dot(Mq, np.dot(dS, nu)) + nq)

    dx = np.zeros_like(x)
    dx[:model.nv] = dq
    dx[model.nv:] = np.linalg.inv(M).dot(tau - n)
    return dx

def ode4(model, data, x0, dt, tau, S, dS):
    # compute runge kutta solutions
    s1 = f(model, data, S, dS, tau, x0)
    x1 = x0 + dt/2*s1

    s2 = f(model, data, S, dS, tau, x1)
    x2 = x0 + dt/2*s2

    s3 = f(model, data, S, dS, tau, x2)
    x3 = x0 + dt*s3

    s4 = f(model, data, S, dS, tau, x3)

    # final solution
    x = x0 + dt*(s1 + 2*s2 + 2*s3 + s4)/6
    return x

def quat_to_rpy(q):    
    # Convert the quaternion to a rotation matrix
    rotation_matrix = pin.Quaternion(q[6], q[3], q[4], q[5]).toRotationMatrix()
    # Convert the rotation matrix to RPY (roll, pitch, yaw)
    rpy = pin.rpy.matrixToRpy(rotation_matrix)

    # Convert the wheel angles to cos and sin
    left_wheel = math.atan2(q[8], q[7])
    right_wheel = math.atan2(q[10], q[9])

    # Return [x, y, z, roll, pitch, yaw, cos(left wheel), cos(right wheel)]
    return np.array([q[0], q[1], q[2], rpy[0], rpy[1], rpy[2], left_wheel, right_wheel])

def rpy_to_quat(q):
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
    