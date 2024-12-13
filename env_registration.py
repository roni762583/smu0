from gymnasium.envs.registration import register

register(
    id="CustomTradingEnv-v0",  # Name of the environment
    entry_point="your_module_name:CustomTradingEnv",  # Path to your environment class
)
