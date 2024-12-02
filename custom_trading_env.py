import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomTradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, action_space_size=4, observation_space_shape=(10,), render_mode=None):
        super().__init__()

        # Action space: Discrete actions (e.g., Buy, Sell, Hold, Close)
        self.action_space = spaces.Discrete(action_space_size)

        # Observation space: Vector of size `observation_space_shape`
        # Allowing values in a realistic range (can be adjusted for trading data specifics)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_space_shape, dtype=np.float32
        )

        # Render mode (currently unused but accepted)
        assert render_mode in self.metadata["render_modes"], f"Invalid render_mode: {render_mode}"
        self.render_mode = render_mode

        # Initialize state and variables
        self.state = None
        self.current_step = 0
        self.max_steps = 100  # Maximum number of steps per episode

    def reset(self, seed=None, options=None):
        # Reset the environment to its initial state
        super().reset(seed=seed)  # Ensure seed initialization for reproducibility
        self.current_step = 0
        self.state = np.random.uniform(-1.0, 1.0, self.observation_space.shape)
        info = {"reset_info": "Environment reset successfully"}
        return self.state, info

    def step(self, action):
        # Update state, calculate reward, and determine if the episode is done
        self.current_step += 1

        # Example reward logic (adjust based on trading objectives)
        if action == 0:  # Assume "Buy"
            reward = np.random.uniform(0, 1)  # Example profit
        elif action == 1:  # Assume "Sell"
            reward = np.random.uniform(-1, 0)  # Example loss
        else:
            reward = 0  # Neutral reward for "Hold" or "Close"

        # Update the state with a dummy logic (replace with trading dynamics)
        self.state = np.random.uniform(-1.0, 1.0, self.observation_space.shape)

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Gymnasium expects step to return (state, reward, done, truncation, info)
        truncation = False  # Add logic for truncation if needed
        info = {"step_info": f"Step {self.current_step}"}

        return self.state, reward, done, truncation, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, State: {self.state}")
        else:
            pass  # No rendering for other modes

    def close(self):
        # Perform any necessary cleanup
        print("Environment closed.")


# Register the custom environment with Gymnasium
from gymnasium.envs.registration import register

register(
    id="CustomTradingEnv-v0",  # Unique identifier
    entry_point="custom_trading_env:CustomTradingEnv",  # Module and class name
)
