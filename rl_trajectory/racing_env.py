import gym
import numpy as np
import rclpy
from rclpy.node import Node
from a2rl_bs_msgs.msg import VectornavIns
from vectornav_msgs.msg import ImuGroup
from pynput.keyboard import Key, Controller
import subprocess
import time
import pandas as pd
from scipy.spatial import distance

class RacingEnv(gym.Env, Node):
    def __init__(self, left_bound_csv='rl_trajectory/yasBounds/LeftBound.csv', 
                 right_bound_csv='rl_trajectory/yasBounds/RightBound.csv'):
        super().__init__('racing_env_node')
        
        # Action space: [throttle, steering], each with 3 options
        # Throttle: 0 (off), 1 (short press), 2 (long press)
        # Steering: 0 (off), 1 (short left), 2 (short right)
        self.action_space = gym.spaces.MultiDiscrete([3, 3])

        # Observation space: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, accel_x, accel_y, accel_z, yaw]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                               shape=(10,), dtype=np.float32)

        # ROS2 subscribers
        self.ins_sub = self.create_subscription(VectornavIns, '/a2rl/vn/ins', self.ins_callback, 10)
        self.imu_sub = self.create_subscription(ImuGroup, '/vectornav/raw/imu', self.imu_callback, 10)

        # State variables
        self.state = np.zeros(10)
        self.last_position = np.zeros(3)
        self.lap_count = 0
        self.last_lap_time = time.time()
        self.done = False

        # Keyboard controller
        self.keyboard = Controller()

        # Load track bounds
        self.left_bound = pd.read_csv(left_bound_csv, header=None, names=['x', 'y', 'z']).to_numpy()
        self.right_bound = pd.read_csv(right_bound_csv, header=None, names=['x', 'y', 'z']).to_numpy()

        # Start/finish line
        self.start_finish_line = {'x': -20.0, 'y': -680.0, 'tolerance': 5.0}
        self.track_width_threshold = 15.0

    def ins_callback(self, msg):
        self.state[0] = msg.position_enu_ins.x
        self.state[1] = msg.position_enu_ins.y
        self.state[2] = msg.position_enu_ins.z
        self.state[3] = msg.velocity_enu_ins.x
        self.state[4] = msg.velocity_enu_ins.y
        self.state[5] = msg.velocity_enu_ins.z
        self.state[9] = msg.orientation_ypr.z

    def imu_callback(self, msg):
        self.state[6] = msg.accel.x
        self.state[7] = msg.accel.y
        self.state[8] = msg.accel.z

    def reset(self):
        subprocess.run(["simcli", "--reset-to-track"], check=True)
        self.done = False
        self.lap_count = 0
        self.last_lap_time = time.time()
        self.last_position = self.state[:3].copy()
        
        self.keyboard.press('k')
        self.keyboard.release('k')
        time.sleep(2)
        rclpy.spin_once(self, timeout_sec=0.1)
        return self.state

    def step(self, action):
        throttle, steering = action
        
        # Throttle actions
        if throttle == 1:  # Short accelerate
            self.keyboard.press(Key.up)
            time.sleep(0.1)  # Light press
            self.keyboard.release(Key.up)
        elif throttle == 2:  # Long accelerate
            self.keyboard.press(Key.up)
            time.sleep(0.3)  # Stronger press
            self.keyboard.release(Key.up)
        # Note: Down (brake) will be separate to allow combos

        # Steering actions (can happen at the same time as throttle)
        if steering == 1:  # Short left
            self.keyboard.press(Key.left)
            time.sleep(0.1)
            self.keyboard.release(Key.left)
        elif steering == 2:  # Short right
            self.keyboard.press(Key.right)
            time.sleep(0.1)
            self.keyboard.release(Key.right)

        # Brake actions (can combine with steering)
        if throttle == 2 and steering in [0, 1, 2]:  # Long press includes braking option
            self.keyboard.press(Key.down)
            time.sleep(0.3)  # Stronger brake
            self.keyboard.release(Key.down)
        elif throttle == 1 and steering in [0, 1, 2]:  # Short brake if no accel
            self.keyboard.press(Key.down)
            time.sleep(0.1)  # Light brake
            self.keyboard.release(Key.down)

        rclpy.spin_once(self, timeout_sec=0.1)
        reward = self._compute_reward()
        self.done = self._check_done()
        self.last_position = self.state[:3].copy()

        return self.state, reward, self.done, {}

    def _compute_reward(self):
        reward = 0.0
        current_pos = self.state[:2]
        start_pos = np.array([self.start_finish_line['x'], self.start_finish_line['y']])
        dist_to_start = np.linalg.norm(current_pos - start_pos)

        if dist_to_start < self.start_finish_line['tolerance']:
            if self.lap_count == 0 or time.time() - self.last_lap_time > 5.0:
                lap_time = time.time() - self.last_lap_time
                if lap_time > 0:
                    reward += 1000.0 / lap_time
                    self.lap_count += 1
                    self.last_lap_time = time.time()

        speed = np.sqrt(self.state[3]**2 + self.state[4]**2)
        reward += speed * 0.1

        if self._is_off_track():
            reward -= 100.0
        if self.state[3] < 0:  # Still penalize if somehow moving backward
            reward -= 50.0
        if abs(self.state[7]) > 5.0:
            reward -= 20.0
        if speed < 0.1:
            reward -= 10.0

        return reward

    def _check_done(self):
        return self._is_off_track() or self.lap_count >= 5

    def _is_off_track(self):
        current_pos = self.state[:2]
        dist_to_left = np.min(distance.cdist([current_pos], self.left_bound[:, :2]))
        dist_to_right = np.min(distance.cdist([current_pos], self.right_bound[:, :2]))
        return dist_to_left > self.track_width_threshold and dist_to_right > self.track_width_threshold

if __name__ == "__main__":
    rclpy.init()
    env = RacingEnv()
    obs = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
        if done:
            obs = env.reset()
    env.destroy_node()
    rclpy.shutdown()