#!/usr/bin/env python

import math

#Pinocchio libraries
import pinocchio as pin
from pinocchio.utils import *
import numpy as np
import matplotlib.pyplot as plt
import os

DEBUG = False

def normalize_angle(angle):
        """Normalizza un angolo all'intervallo [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi            
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    

# Controller class
class Controller:
    def __init__(self):

        # Define controller parameters
        self.wheel_distance = 0.4044 
        self.wheel_radius = 0.0985  
        self.v_r = 0.08
        self.w_r = 0.1

        self.k1 = 1
        self.k2 = 3
        self.k3 = 1 
        self.k4 = 5
        self.k5 = 5

        self.urdfFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tiago.urdf')
        self.rmodel = pin.buildModelFromUrdf(self.urdfFile, pin.JointModelFreeFlyer())
        self.rdata = self.rmodel.createData()

        # Initialize robot configuration and velocity
        self.q = pin.neutral(self.rmodel)
        self.v = pin.utils.zero(self.rmodel.nv)

        # Define inertia matrix
        self.M = np.array([])
        
        # Initialize desired trajectory variables
        self.thetad = 0
        self.xd = 0
        self.yd = 0 

        # Initialize some lists to store data to plot
        self.desired_trajectory = []
        self.actual_trajectory = []
        self.torques = []
        self.slipping = []
        self.slipping_est = []

        self.iL = 0
        self.iR = 0
 

    # Callback function for the estimated state subscriber
    def command(self, msg):

        dt = 0.1

        # Process the received state
        extended_state = msg  # [x, y, theta, iL, iR, omegaLe, omegaRe]
        if DEBUG: ("stato: ", extended_state[:3])

        # Print the formatted list
        if DEBUG: print("Extended state: ",[format(x, f".{3}f") for x in extended_state])

        # Extracting x and y coordinates, and tauL and tauR
        theta = extended_state[2]
        iLe = extended_state[5]
        iRe = extended_state[6]
        if DEBUG: print(iLe)

        self.slipping.append(np.array([self.iL,self.iR]).copy())
        self.slipping_est.append(np.array([iLe,iRe]))
        
        # Compute desired trajectory
        self.thetad += dt * self.w_r
        self.thetad = normalize_angle(self.thetad)
        self.xd += dt * self.v_r * np.cos(self.thetad)
        self.yd += dt * self.v_r * np.sin(self.thetad)


        #TODO: ########################### HERE IMPLEMENT 10,11,12,13 ##############################
        desired_configuration = np.array([self.xd, self.yd, self.thetad])
        error = desired_configuration - extended_state[:3]
        error[2] = normalize_angle(error[2])
        if DEBUG: print("thetad: ",self.thetad,"theta: ",extended_state[2], "error on theta: ", error[2])

        self.desired_trajectory.append(desired_configuration)
        self.actual_trajectory.append(extended_state[:3])
        
        # Compute e = (e1, e2, e3).T
        e = np.array([
            [math.cos(theta), math.sin(theta), 0],
            [-math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]]) @ error.T
        
        # Compute omega and v
        omega = self.w_r + self.v_r / 2 * (self.k3 * (e[1] + self.k3 * e[2]) + (math.sin(e[2]) / self.k2))
        v = self.v_r * math.cos(e[2]) - self.k3 * e[2] * omega + self.k1 * e[0]
        if DEBUG: print("e3: ", e[2])
        if DEBUG: print("velocities: ", [format(x, f".{3}f") for x in [v, omega]])

        # Compute e_dot = (e1_dot, e2_dot, e3_dot)
        e_dot = np.array(
            [omega * e[1] + self.v_r * math.cos(e[2]) - v,
            -omega * e[0] + self.v_r * math.sin(e[2]),
            self.w_r - omega])

        # Compute omega_dot and v_dot
        '''
        omega_dot = self.v_r / 2 * (self.k3 * (e_dot[1] + self.k3 * e_dot[2]) + (math.cos(e[2])*e_dot[2] / self.k2))
        v_dot = - self.v_r * math.sin(e[2]) * e_dot[2] - self.k3 * e_dot[2] * omega - self.k3 * e[2] * omega_dot + self.k1 * e_dot[0]
        '''

        # Compute ξd = T @ (v, omega).T
        T = self.wheel_radius / (2 * self.wheel_distance) * np.array([
            [self.wheel_distance * (1 - iLe), self.wheel_distance * (1 - iRe)],
            [-2 * (1 - iLe), 2 * (1 - iRe)]])
        T_inv = np.linalg.inv(T)
        
        omegaLd = T_inv[0][0] * v + T_inv[0][1] * omega
        omegaRd = T_inv[1][0] * v + T_inv[1][1] * omega

        # Compute ξd_dot
        omegaLd_dot = 0#T_inv[0] @ np.array([v_dot, omega_dot]).T + T_inv_dot[0] @ np.array([v, omega]).T
        omegaRd_dot = 0#_inv[1] @ np.array([v_dot, omega_dot]).T + T_inv_dot[1] @ np.array([v, omega]).T


        if DEBUG: print("Desired velocities: ", [format(x, f".{3}f") for x in [omegaLd, omegaRd]])

        #TODO: ########################### EQUATIONS 7,4,3 ###################################
        omegaLe = extended_state[3]
        omegaRe = extended_state[4]

        u = np.array([omegaLd_dot, omegaRd_dot]) - np.array([self.k4*(omegaLe-omegaLd), self.k5*(omegaRe-omegaRd)])

        S = np.array([
            [self.wheel_distance * self.wheel_radius * (1 - iLe) * math.cos(theta), self.wheel_distance * self.wheel_radius * (1 - iRe) * math.cos(theta)],
            [self.wheel_distance * self.wheel_radius * (1 - iLe) * math.sin(theta), self.wheel_distance * self.wheel_radius * (1 - iRe) * math.sin(theta)],
            [-2*self.wheel_radius * (1 - iLe), 2*self.wheel_radius * (1 - iRe)]
            ]) * (1 / (2 * self.wheel_distance))
        
        B = np.array([
            [math.cos(theta),math.cos(theta)],
            [math.sin(theta),math.sin(theta)],
            [-self.wheel_distance/2, self.wheel_distance/2]
            ])

        # Compute all the terms of the dynamic model
        pin.computeAllTerms(self.rmodel, self.rdata, self.q, self.v)

        # Extract centroidal mass and centroidal inertia matrix
        m = self.rdata.Ig.mass                  # Total mass of the robot
        I_zz = self.rdata.Ig.inertia[2, 2]      # Moment of inertia about the vertical axis    

        # Compute the 3x3 mass matrix M
        self.M = np.array([[m, 0, 0], [0, m, 0], [0, 0, I_zz]])
        # Compute Bbar
        Bbar = S.T @ B

        # Aggiunta della regolarizzazione a Bbar
        #lambda_reg = 1e-4  # Può essere necessario aggiustare questo valore
        #Bbar_reg = Bbar + lambda_reg * np.eye(Bbar.shape[0])

        # Calcolo dell'inversa di Bbar regolarizzata
        #Bbar_inv = np.linalg.inv(Bbar_reg)

        Mbar = S.T @ self.M @ S
        tau = (np.linalg.inv(Bbar) @ Mbar) @ u.T
        #tau = (np.linalg.pinv(Bbar)@ Mbar) @ u.T
        # Calcolo di tau con la regolarizzazione
        #tau = Bbar_inv @ Mbar @ u.T 

        # Append torques
        self.torques.append(tau)

        return np.array(tau)

        #print("Torques: ", [format(x, f".{3}f") for x in tau])

        # We have to invert the relation 

        left_joint_gain = 30
        right_joint_gain = 30

        omegaL = omegaLe + tau[0]/left_joint_gain 
        omegaR = omegaRe + tau[1]/right_joint_gain 
        

        T = self.wheel_radius / (2 * self.wheel_distance) * np.array([
            [self.wheel_distance * (1 - self.iL), self.wheel_distance * (1 - self.iR)],
            [-2 * (1 - self.iL), 2 * (1 - self.iR)]])

        ##################################################################################################################################
        # Calculate linear and angular velocities, this is a plus conversion that we need just to build the ROS message

        linear_velocity = T[0][0] * omegaL + T[0][1] * omegaR
        angular_velocity = T[1][0] * omegaL + T[1][1] * omegaR

        return np.array([linear_velocity,angular_velocity])
    

    def plot_results(self):
        # Plot the desired and actual trajectories
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot([x[0] for x in self.desired_trajectory], [x[1] for x in self.desired_trajectory], label='Desired trajectory', color='blue')
        ax.plot([x[0] for x in self.actual_trajectory], [x[1] for x in self.actual_trajectory], label='Actual trajectory', color='red')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Desired and actual trajectories')
        ax.set_aspect('equal')
        ax.legend()
        plt.show()

  
