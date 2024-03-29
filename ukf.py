#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pinocchio as pin

DEBUG = False

def normalize_angle(angle):
        """Normalizza un angolo all'intervallo [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi            
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
class UKF:
    def __init__(self, ):
        
        self.state_dim = 7  # State dimension: [x, y, theta, omegaL, omegaR, iL, iR]
        self.measurement_dim = 5  # Measurement dimension: [x, y, theta, omegaL, omegaR]

        # Process and measurement noise covariances
        self.Q = np.eye(self.state_dim) * 0.001  # Adjust as needed
        self.R = np.eye(self.measurement_dim) * 0.001  # Adjust as needed

        # State vector and covariance matrix
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim)

        self.m = np.zeros(self.measurement_dim)

        # UKF parameters
        self.alpha = 1
        self.beta = 2
        self.kappa = 1
        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim

        # Weights for sigma points
        self.Wm = np.full(2 * self.state_dim + 1, 1 / (2 * (self.state_dim + self.lambda_)))
        self.Wc = np.full(2 * self.state_dim + 1, 1 / (2 * (self.state_dim + self.lambda_)))
        self.Wm[0] = self.lambda_ / (self.state_dim + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)


        self.tau = np.array([0,0])

        # Wheel parameters
        self.wheel_distance = 0.2022  
        self.wheel_radius = 0.0985 

        self.urdfFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tiago.urdf')
        self.rmodel = pin.buildModelFromUrdf(self.urdfFile, pin.JointModelFreeFlyer())
        self.rdata = self.rmodel.createData()
        self.q = pin.neutral(self.rmodel)
        self.v = pin.utils.zero(self.rmodel.nv)
        

    def read_measures(self, x_robot,tau):
        self.m[0] = x_robot[0]      # x
        self.m[1] = x_robot[1]      # y
        self.m[2] = x_robot[2]      # theta
        self.m[3] = x_robot[3]      # omegaL
        self.m[4] = x_robot[4]      # omegaR
        self.tau = tau

    def compute_sigma_points(self):
        # Calculate square root of P matrix using Singular Value Decomposition (SVD)
        U, S, V = np.linalg.svd(self.P)
        sqrt_P = np.dot(U, np.diag(np.sqrt(S)))
        
        sigma_points = np.zeros((2 * self.state_dim + 1, self.state_dim))
        sigma_points[0] = self.x
        
        # Calculation of sigma points using the square root of the covariance matrix
        sqrt_matrix = np.sqrt(self.state_dim + self.lambda_) * sqrt_P
        for k in range(self.state_dim):
            sigma_points[k + 1] = self.x + sqrt_matrix[:, k]
            sigma_points[self.state_dim + k + 1] = self.x - sqrt_matrix[:, k]
        #print(sigma_points)
        
        return sigma_points


    def predict(self, dt):
        # Predict the state using the UKF equations
        sigma_points = self.compute_sigma_points()
        for i, point in enumerate(sigma_points):
            sigma_points[i] = self.f(point, dt)

        # Compute the predicted state mean
        x_pred = np.sum(self.Wm[:, np.newaxis] * sigma_points, axis=0)
        x_pred[2] = normalize_angle(x_pred[2]) #Normalize between -pi and pi
        
        # Compute the predicted state covariance
        P_pred = self.Q.copy()
        for i in range(2 * self.state_dim + 1):
            y = sigma_points[i] - x_pred
            P_pred += self.Wc[i] * np.outer(y, y)
        
        # Update the state and covariance with the prediction
        self.x = x_pred
        #print(self.P)
        self.P = P_pred

    def h(self, x):
        return np.array([x[0], x[1], x[2], x[3], x[4]])
    
    
    def f(self, x, dt):
        """
        Modello di processo che descrive come lo stato del sistema evolve nel tempo.
        
        :param x: Stato corrente del sistema.
        :param dt: Incremento temporale.
        :return: Stato aggiornato.
        """

        theta = self.x[2]
        iLe = self.x[5]
        iRe = self.x[6]

        S = (1 / (2 * self.wheel_distance)) * np.array([
            [self.wheel_distance * self.wheel_radius * (1 - iLe) * math.cos(theta), self.wheel_distance * self.wheel_radius * (1 - iRe) * math.cos(theta)],
            [self.wheel_distance * self.wheel_radius * (1 - iLe) * math.sin(theta), self.wheel_distance * self.wheel_radius * (1 - iRe) * math.sin(theta)],
            [-2*self.wheel_radius * (1 - iLe), 2*self.wheel_radius * (1 - iRe)]
        ]) 
        
        B = np.array([
            [math.cos(theta), math.cos(theta)],
            [math.sin(theta), math.sin(theta)],
            [-self.wheel_distance/2, self.wheel_distance/2]
        ])
        
        pin.computeAllTerms(self.rmodel, self.rdata, self.q, self.v)
        # Extract centroidal mass and centroidal inertia matrix
        m = self.rdata.Ig.mass                  # Total mass of the robot
        I_zz = self.rdata.Ig.inertia[2, 2]      # Moment of inertia about the vertical axis    

        # Compute the 3x3 mass matrix M
        M = np.array([[m, 0, 0], [0, m, 0], [0, 0, I_zz]])
        
        Bbar = S.T @ B
        Mbar = S.T @ M @ S
        

        csidot = np.linalg.inv(Mbar) @ Bbar @ self.tau       # [w_l_dot, w_r_dot]


        omegaL = csidot[0] * dt
        omegaR = csidot[1] * dt

        x[3] += omegaL
        x[4] += omegaR

        T = self.wheel_radius / (2 * self.wheel_distance) * np.array([
            [self.wheel_distance * (1 - iLe), self.wheel_distance * (1 - iRe)],
            [-2 * (1 - iLe), 2 * (1 - iRe)]])
   

        velocities = T @ np.array([x[3],x[4]])
        v = velocities[0]
        omega = velocities[1]

        
        # Aggiorna la posizione e l'angolo in base al modello di movimento del robot.
        x[0] += dt * v * np.cos(x[2])  # Aggiornamento della posizione x
        x[1] += dt * v * np.sin(x[2])  # Aggiornamento della posizione y
        x[2] += dt * omega              # Aggiornamento dell'angolo theta

        #noise = np.random.normal(0,0.001)
        #noiseL = np.random.normal(0,self.P[5][5])
        #noiseR = np.random.normal(0,self.P[6][6])
        #noiseL = np.clip(noiseL,-10,10)
        #noiseR = np.clip(noiseL,-10,10)
        #x[5] += noiseL
        #x[5] = np.clip(x[5],-0.9,0.9)
        #x[6] += noiseR
        #x[6] = np.clip(x[6],-0.9,0.9)
   
        return x

    def update(self):
        # Update the state with the measurement
        sigma_points = self.compute_sigma_points()
        
        # Compute the predicted measurement mean
        Z = np.array([self.h(point) for point in sigma_points])
        z_pred = np.sum(self.Wm[:, np.newaxis] * Z, axis=0)
        
        # Compute the predicted measurement covariance
        P_zz = self.R.copy()
        for i in range(2 * self.state_dim + 1):
            y = Z[i] - z_pred
            P_zz += self.Wc[i] * np.outer(y, y)
        
        # Compute the cross-covariance matrix
        P_xz = np.zeros((self.state_dim, self.measurement_dim))
        for i in range(2 * self.state_dim + 1):
            P_xz += self.Wc[i] * np.outer(sigma_points[i] - self.x, Z[i] - z_pred)
        
        # Compute the Kalman gain
        K = np.dot(P_xz, np.linalg.inv(P_zz))
        
        # Update the state with the measurement
        measurement = self.m
        self.x += np.dot(K, measurement - z_pred)
        self.P -= np.dot(K, np.dot(P_zz, K.T))






