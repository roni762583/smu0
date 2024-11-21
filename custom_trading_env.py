import gym
from gym import spaces
import numpy as np

# Define your custom environment
class CustomTradingEnv(gym.Env):
    def __init__(self, action_space_size=4, observation_space_shape=(10,)):
        super().__init__()
        self.action_space = spaces.Discrete(action_space_size)  # Define 4 actions (e.g., Buy, Sell, Hold, Close)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=observation_space_shape, dtype=np.float32)
        self.state = np.zeros(observation_space_shape)
        self.current_step = 0

    def step(self, action):
        # Define how the environment responds to actions
        self.current_step += 1
        reward = 1.0 if action == 0 else -1.0  # Dummy reward logic
        self.state = np.random.randn(self.observation_space.shape[0])  # Dummy state update
        done = self.current_step >= 100
        return self.state, reward, done, {}

    def reset(self):
        # Reset the environment
        self.current_step = 0
        self.state = np.zeros(self.observation_space.shape[0])
        return self.state

# Register your custom environment with Gym
gym.envs.registration.register(
    id='CustomTrading-v0',  # Unique identifier
    entry_point='__main__:CustomTradingEnv',  # The entry point to the environment class
)
