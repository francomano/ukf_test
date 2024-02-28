import numpy as np
import pinocchio

def f(model, data, S, dS, tau, x):
    q = x[:model.nv]
    q_pin = qnv2pinocchio(q, model.nq)
    nu = x[model.nv:]

    semi_axle = 0.2022
    wheel_radius = 0.0985
    theta = q[5]

    # null space Matrix
    S[0, 0] = wheel_radius*np.cos(theta)/2
    S[0, 1] = wheel_radius*np.cos(theta)/2
    S[1, 0] = wheel_radius*np.sin(theta)/2
    S[1, 1] = wheel_radius*np.sin(theta)/2
    S[5, 0] = -wheel_radius/(2*semi_axle)
    S[5, 1] = wheel_radius/(2*semi_axle)
    S[model.nv:, :] = np.eye(len(nu))

    dq = np.dot(S, nu)
    dtheta = dq[5]

    # derivative of null space matrix
    dS[0, 0] = -dtheta*wheel_radius*np.sin(theta)/2
    dS[0, 1] = -dtheta*wheel_radius*np.sin(theta)/2
    dS[1, 0] = dtheta*wheel_radius*np.cos(theta)/2
    dS[1, 1] = dtheta*wheel_radius*np.cos(theta)/2

    Mq = pinocchio.crba(model, data, q_pin)
    data.M[np.tril_indices(model.nv, -1)] = data.M.transpose()[np.tril_indices(model.nv, -1)]
    Mq = data.M
    Damp = np.eye(model.nv)
    for i in range(model.nv):
        Damp[i, i] = model.damping[i]
    nq = pinocchio.rnea(model, data, q_pin, dq, np.zeros(model.nv)) + np.dot(Damp, dq)

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



def qnv2pinocchio(q, nq):
    """
    Convert the floating-point base variables from RPY to quaternion representation.
    
    Parameters:
    - q: The input state vector including RPY angles.
    - nq: The total number of generalized coordinates, including the quaternion.
    
    Returns:
    - q_pin: The state vector with the base orientation represented as a quaternion.
    """

    print("q: ", q)
    print("nq: ", nq)

    return q

    # Assuming the first 3 elements are position and the next 3 are RPY angles
    pos = q[:3]  # Extract position
    rpy = q[3:6].astype(float)  # Ensure RPY angles are floats
    
    # Convert RPY to SE3 object (transformation matrix), then extract the quaternion
    se3 = pinocchio.SE3.Identity()
    se3.rotation = pinocchio.rpyToMatrix(rpy[0], rpy[1], rpy[2])  # Convert RPY to rotation matrix
    quaternion = pinocchio.Quaternion(se3.rotation).coeffs()  # Convert rotation matrix to quaternion
    
    # Construct the q_pin vector
    q_pin = np.zeros(nq)
    q_pin[:3] = pos  # Set position
    q_pin[3:7] = quaternion  # Set quaternion (note: the order is [x, y, z, w])
    
    # If there are additional joints beyond the base orientation, copy them over
    if len(q) > 6:
        q_pin[7:] = q[6:]
    
    return q_pin