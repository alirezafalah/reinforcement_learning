import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from racing_env import RacingEnv

# Optional: WandB callback for logging
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self):
        wandb.log({
            "reward": self.locals["rewards"][0],
            "lap_time": self.training_env.get_attr("lap_time")[0],
            "episode_length": self.n_steps
        })
        return True

# Initialize ROS2 and environment
rclpy.init()
env = RacingEnv()

# Optional: Initialize WandB
wandb.init(project="racing-rl-keybinds", entity="your_username", config={
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "algorithm": "PPO"
}, mode="online")  # Set mode="disabled" to skip WandB

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)

# Train the agent
callback = WandbCallback() if wandb.run else None
model.learn(total_timesteps=100000, callback=callback)

# Save the model
model.save("ppo_racing_agent")
if wandb.run:
    wandb.save("ppo_racing_agent.zip")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    if done:
        obs = env.reset()

# Cleanup
env.destroy_node()
rclpy.shutdown()
if wandb.run:
    wandb.finish()