import ode4
import controller
import ukf
import numpy as np
import os
import pinocchio as pin

DEBUG = True

# Initialize the robot model and data 
urdfFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tiago.urdf')
full_model = pin.buildModelFromUrdf(urdfFile, pin.JointModelFreeFlyer())

# Base-related joints to keep
base_joints_to_keep = [
    'universe',             # Must be kept
    'root_joint',           # Floating base
    'wheel_left_joint',
    'wheel_right_joint',
]

# Get the IDs of all joints to keep
joints_to_keep_ids = [full_model.getJointId(joint) for joint in base_joints_to_keep if full_model.existJointName(joint)]

# Get the IDs of all joints to lock (all joints except the ones to keep)
joints_to_lock_ids = [jid for jid in range(full_model.njoints) if jid not in joints_to_keep_ids]

# Set initial position for the joints we want to keep; the rest will be locked
initial_joint_config = np.zeros(full_model.nq)

# Build the reduced model
model = pin.buildReducedModel(full_model, joints_to_lock_ids, initial_joint_config)

# Check dimensions of the reduced model
print('Reduced model', model)

# Create the data for the reduced model
data = model.createData()

# Initialize robot configuration and velocity
q = pin.neutral(model)
v = pin.utils.zero(model.nv)

print("MODEL JOINTS", model.joints[0])

pin.updateFramePlacements(model, data)
print("DATAAA", pin.getFrameVelocity(model, data, model.getFrameId('root_joint')))

# Define the controller and UKF objects
contr = controller.Controller()
ukf_est = ukf.UKF()

# Starting configuration and initial state
starting_conf = [0,0,0,0,0,0,0]                 # [x, y, theta, iL, iR, omegaLe, omegaRe]
#x0 = np.array(starting_conf + [0,0,0,0,0,0,0])  # Assuming initial velocities are zero

# Null space matrix and its derivative
#S = np.zeros((model.nv + len(starting_conf), 2))  # Adjust the size accordingly
#dS = np.zeros_like(S)

# Time step
dt = 0.1  # Example time step

# Loop for the simulation
for i in range(100):
    # Compute the torques
    tau = contr.command(starting_conf)
    if DEBUG: print("Torques", tau)

    # Update the state with the ode4 integration
    #x0 = ode4.ode4(model, data, x0, dt, tau, S, dS)
    #if DEBUG: print("ODE4", x0)
    
    # Update starting_conf for the next iteration
    #starting_conf = x0[:len(starting_conf)]
    
    #ukf.read_measures() #must be implemented
    #ukf.compute_sigma_points()

    if DEBUG: print("UKF", ukf_est.x)

    break

#contr.plot_results()
