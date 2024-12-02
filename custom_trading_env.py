import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomTradingEnv(gym.Env):
    metadata = {"render_modes": [None, "human", "rgb_array"]}

    def __init__(self, action_space_size=4, observation_space_shape=(10,), render_mode=None):
        super().__init__()
        self.render_mode = render_mode  # Allow None as default
        if self.render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render_mode: {self.render_mode}")

        # Define action and observation spaces
        self.action_space = spaces.Discrete(action_space_size)  # 4 actions (e.g., Buy, Sell, Hold, Close)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_space_shape, dtype=np.float32
        )

        # Initialize state and variables
        self.state = None
        self.current_step = 0
        self.max_steps = 100  # Maximum steps in an episode

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)  # Seed initialization for reproducibility
        self.current_step = 0
        self.state = np.random.uniform(-1.0, 1.0, self.observation_space.shape)
        info = {"reset_info": "Environment reset successfully"}
        return self.state, info

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        self.current_step += 1

        # Example reward logic (customize as needed)
        if action == 0:  # Assume "Buy"
            reward = np.random.uniform(0, 1)  # Example profit
        elif action == 1:  # Assume "Sell"
            reward = np.random.uniform(-1, 0)  # Example loss
        else:
            reward = 0  # Neutral reward for "Hold" or "Close"

        # Update the state (dummy example logic, replace with trading rules)
        self.state = np.random.uniform(-1.0, 1.0, self.observation_space.shape)

        # Check if the episode is done
        done = self.current_step >= self.max_steps
        truncation = False  # Add truncation logic if needed

        info = {"step_info": f"Step {self.current_step}"}

        return self.state, reward, done, truncation, info

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, State: {self.state}")
        elif self.render_mode == "rgb_array":
            # Replace with a visualization method if applicable
            return np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy RGB array
        else:
            pass  # No rendering for other modes

    def close(self):
        """
        Clean up resources when the environment is closed.
        """
        print("Environment closed.")

'''
# Register the custom environment with Gymnasium
from gymnasium.envs.registration import register

register(
    id="CustomTradingEnv-v0",  # Unique identifier
    entry_point="custom_trading_env:CustomTradingEnv",  # Module and class name
)
'''