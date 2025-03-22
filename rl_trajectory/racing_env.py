import gym
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32  # Assuming /lap_times uses Float32
from pynput.keyboard import Key, Controller
import time

class RacingEnv(gym.Env, Node):
    def __init__(self):
        super().__init__('racing_env_node')
        
        # Action space: [throttle, steering] as continuous values mapped to key presses
        self.action_space = gym.spaces.Box(low=np.array([0.0, -1.0]), 
                                          high=np.array([1.0, 1.0]), 
                                          dtype=np.float32)  # throttle [0,1], steering [-1,1]

        # Observation space: [gps_lat, gps_lon, imu_x, imu_y, imu_z, lap_time]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                               shape=(6,), dtype=np.float32)

        # ROS2 subscribers
        self.gps_sub = self.create_subscription(NavSatFix, '/vectornav/raw/gps', self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/vectornav/raw/imu', self.imu_callback, 10)
        self.lap_sub = self.create_subscription(Float32, '/lap_times', self.lap_callback, 10)

        # State variables
        self.state = np.zeros(6)  # [gps_lat, gps_lon, imu_x, imu_y, imu_z, lap_time]
        self.lap_time = 0.0
        self.last_lap_time = 0.0
        self.done = False

        # Keyboard controller
        self.keyboard = Controller()

    def gps_callback(self, msg):
        self.state[0] = msg.latitude
        self.state[1] = msg.longitude

    def imu_callback(self, msg):
        self.state[2] = msg.linear_acceleration.x
        self.state[3] = msg.linear_acceleration.y
        self.state[4] = msg.linear_acceleration.z

    def lap_callback(self, msg):
        self.state[5] = msg.data
        self.lap_time = msg.data

    def reset(self):
        # Reset the simulation (assumes manual reset or sim auto-resets)
        self.done = False
        self.lap_time = 0.0
        self.last_lap_time = 0.0
        
        # Press 'k' to enable keyboard control mode (if needed)
        self.keyboard.press('k')
        self.keyboard.release('k')
        time.sleep(1)  # Allow sim to reset
        return self.state

    def step(self, action):
        # Apply action (throttle, steering) via keyboard
        throttle, steering = action
        
        # Map throttle to forward/backward keys
        if throttle > 0.1:  # Forward
            self.keyboard.press(Key.up)
            time.sleep(0.05 * throttle)  # Simulate throttle intensity
            self.keyboard.release(Key.up)
        elif throttle < -0.1:  # Backward (brake)
            self.keyboard.press(Key.down)
            time.sleep(0.05 * abs(throttle))
            self.keyboard.release(Key.down)

        # Map steering to left/right keys
        if steering > 0.1:  # Right
            self.keyboard.press(Key.right)
            time.sleep(0.05 * steering)
            self.keyboard.release(Key.right)
        elif steering < -0.1:  # Left
            self.keyboard.press(Key.left)
            time.sleep(0.05 * abs(steering))
            self.keyboard.release(Key.left)

        # Spin ROS2 node to update state
        rclpy.spin_once(self, timeout_sec=0.1)

        # Compute reward and check termination
        reward = self._compute_reward()
        self.done = self._check_done()

        return self.state, reward, self.done, {}

    def _compute_reward(self):
        # Reward: Faster lap time = higher reward
        if self.lap_time > 0 and self.last_lap_time != self.lap_time:
            reward = 1000.0 / self.lap_time  # Inverse of lap time
            self.last_lap_time = self.lap_time
        else:
            reward = 0.0

        # Penalties
        if self._is_off_track():  # Implement based on GPS bounds
            reward -= 100.0
        if self.state[2] < 0:  # Moving backward
            reward -= 50.0
        if abs(self.state[3]) > 5.0:  # Slipping (high lateral acceleration)
            reward -= 20.0
        if abs(self.state[2]) < 0.1:  # Stopped
            reward -= 10.0

        return reward

    def _check_done(self):
        # Episode ends if off-track or lap completed
        return self._is_off_track() or (self.lap_time > 0 and self.lap_time != self.last_lap_time)

    def _is_off_track(self):
        # Define track bounds using GPS coordinates (placeholder)
        # Example: return True if outside bounds
        return False  # Implement based on your track

# Initialize ROS2 (can be imported and run elsewhere)
if __name__ == "__main__":
    rclpy.init()
    env = RacingEnv()
    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
        if done:
            obs = env.reset()
    env.destroy_node()
    rclpy.shutdown()