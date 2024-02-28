import ode4
import controller
import ukf
import numpy as np
import os
import pinocchio as pin

DEBUG = True

# Initialize the robot model and data 
urdfFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tiago.urdf')
model = pin.buildModelFromUrdf(urdfFile, pin.JointModelFreeFlyer())
data = model.createData()

print("MODEL", model.nv)   

# Define the controller and UKF objects
contr = controller.Controller()
ukf_est = ukf.UKF()

# Starting configuration and initial state
starting_conf = [0,0,0,0,0,0,0]                 # [x, y, theta, iL, iR, omegaLe, omegaRe]
x0 = np.array(starting_conf + [0,0,0,0,0,0,0])  # Assuming initial velocities are zero

# Null space matrix and its derivative
S = np.zeros((model.nv + len(starting_conf), 2))  # Adjust the size accordingly
dS = np.zeros_like(S)

# Time step
dt = 0.1  # Example time step

# Loop for the simulation
for i in range(100):
    # Compute the torques
    tau = contr.command(starting_conf)
    if DEBUG: print("Torques", tau)

    # Update the state with the ode4 integration
    x0 = ode4.ode4(model, data, x0, dt, tau, S, dS)

    if DEBUG: print("ODE4", x0)
    
    # Update starting_conf for the next iteration
    starting_conf = x0[:len(starting_conf)]
    
    #ukf.read_measures() #must be implemented
    #ukf.compute_sigma_points()

    if DEBUG: print("UKF", ukf.x)

    break

#contr.plot_results()
