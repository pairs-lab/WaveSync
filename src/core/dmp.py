import json
import os
import pickle
import numpy as np
import sys
import time
import random
from scipy.signal import savgol_filter
import pybullet as pb
import pybullet_data

# Thêm src vào sys.path để import cấu hình
sys.path.append("./src")
try:
    from utils.RobotConfig import RobotConfig
    from utils.types import RobotType
except ImportError:
    print("Warning: Could not find utils module. Please check the path.")

def get_emotion_params(emotion):
    emo = emotion.lower().strip()
    
    p_rand = random.uniform(0.01, 0.03) 
    
    if emo in ['happy', 'confident', 'surprised']:
        p_exa = random.uniform(1.1, 1.3)
        p_time = random.uniform(0.75, 0.85)
        p_ant = random.uniform(0.08, 0.12)
        spatial_offset = random.uniform(0.05, 0.15)
        
    elif emo == 'sad':
        p_exa = random.uniform(0.6, 0.8)
        p_time = random.uniform(1.2, 1.4)
        p_ant = random.uniform(0.02, 0.05)
        spatial_offset = random.uniform(-0.15, -0.05)
        
    else:
        p_exa = random.uniform(0.9, 1.1)
        p_time = random.uniform(0.9, 1.1)
        p_ant = random.uniform(0.07, 0.1)    
        spatial_offset = random.uniform(-0.05, 0.05)

    return p_exa, p_time, p_ant, p_rand, spatial_offset

class ExpressiveDMP:
    def __init__(self, n_bfs=100, alpha_y=25.0):
        self.n_bfs = n_bfs
        self.alpha_y = alpha_y
        self.beta_y = alpha_y / 4.0
        self.model = None

    def train(self, trajectory, dt, joint_names):
        n_frames, n_joints = trajectory.shape
        tau = n_frames * dt
        y = trajectory
        dy = np.gradient(y, axis=0) / dt
        ddy = np.gradient(dy, axis=0) / dt
        importance = np.max(trajectory, axis=0) - np.min(trajectory, axis=0)
        
        x = np.linspace(1, 0, n_frames)
        centers = np.linspace(1, 0, self.n_bfs)
        widths = np.ones(self.n_bfs) * (self.n_bfs**1.5) 
        
        g = y[-1] 
        f_target = (tau**2) * ddy - self.alpha_y * (self.beta_y * (g - y) - tau * dy)
        
        weights = np.zeros((n_joints, self.n_bfs))
        for j in range(n_joints):
            psi = np.exp(-widths * (x[:, None] - centers)**2)
            for b in range(self.n_bfs):
                weights[j, b] = np.sum(psi[:, b] * x * f_target[:, j]) / (np.sum(psi[:, b] * x**2) + 1e-8)
                
        self.model = {
            "weights": weights, "centers": centers, "widths": widths,
            "g": g, "y0": y[0], "tau_orig": tau, "dt_orig": dt,
            "joint_names": joint_names, "n_frames_orig": n_frames,
            "importance": importance
        }

    def generate(self, p_exa=1.0, p_time=1.0, p_ant=0.0, p_rand=0.0, t_ant_ratio=0.1, target_fps=None):
        if self.model is None: return None
        w_orig = self.model['weights']
        names = self.model['joint_names']
        n_joints = w_orig.shape[0]
        
        w = w_orig.copy()
        if p_rand > 0.0:
            for j in range(n_joints):
                mean_abs_w = np.mean(np.abs(w_orig[j]))
                noise = np.random.normal(0, 1, self.n_bfs)
                w[j] += (1.0 + mean_abs_w) * p_rand * noise
        
        dt = self.model['dt_orig'] if target_fps is None else 1.0 / target_fps
        tau = self.model['tau_orig'] * p_time 
        n_steps = int(tau / dt) 
        if n_steps <= 0: return []
        
        g = self.model['g']
        y = self.model['y0'].copy()
        dy = np.zeros(n_joints)
        new_trajectory = []
        
        importance = self.model['importance']
        important_joints_mask = importance > (np.max(importance) * 0.3)
        t_ant_steps = int(n_steps * t_ant_ratio)
        
        for t in range(n_steps):
            x = 1.0 - (t / n_steps) 
            psi = np.exp(-self.model['widths'] * (x - self.model['centers'])**2)
            f_x = (np.dot(w, psi) / (np.sum(psi) + 1e-8)) * x
            f_x *= p_exa 

            ddy = (self.alpha_y * (self.beta_y * (g - y) - tau * dy) + f_x) / (tau**2)
            
            if t < t_ant_steps and p_ant > 0:
                ddy = np.where(important_joints_mask, -p_ant * ddy, ddy)

            dy += ddy * dt
            y += dy * dt

            frame_dict = {names[i]: float(val) for i, val in enumerate(y)}
            new_trajectory.append(frame_dict)

        return new_trajectory

def setup_pybullet():
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -11.7)
    pb.loadURDF("plane.urdf")
    try:
        robot_config = RobotConfig(RobotType.COMAN)
        robot_id = pb.loadURDF(robot_config.URDF_4_RENDER_PATH)
    except:
        robot_id = pb.loadURDF("r2d2.urdf", [0, 0, 0.5]) 
    pb.changeDynamics(robot_id, -1, mass=0)
    initial_position = [0, 0, 0.53]
    initial_orientation = pb.getQuaternionFromEuler([0, 0, 0])
    pb.resetBasePositionAndOrientation(robot_id, initial_position, initial_orientation)
    pb.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=-15, cameraTargetPosition=initial_position)
    
    num_joints = pb.getNumJoints(robot_id)
    joint_name_to_id = {pb.getJointInfo(robot_id, i)[1].decode("utf-8"): pb.getJointInfo(robot_id, i)[0] for i in range(num_joints)}
    return robot_id, joint_name_to_id

def play_frames(robot_id, joint_name_to_id, frames, fps=30):
    frame_time = 1.0 / fps
    for joi_angles in frames:
        loop_start = time.time()
        for joi_name, angle in joi_angles.items():
            if joi_name in joint_name_to_id:
                joi_index = joint_name_to_id[joi_name]
                pb.resetJointState(robot_id, joi_index, angle)
        pb.stepSimulation()
        elapsed = time.time() - loop_start
        if frame_time > elapsed:
            time.sleep(frame_time - elapsed)
