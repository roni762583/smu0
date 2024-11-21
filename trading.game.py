import numpy as np
import random
import gymnasium as gym
from custom_trading_env import CustomTradingEnv  # Import your custom environment

# Create and use the environment
env = gym.make('CustomTrading-v0')

class Game():
    def __init__(self, 
                 gym_env=None, discount=0.95, limit_of_game_play=float("inf"), 
                 observation_dimension=None, action_dimension=4, 
                 action_map=None, priority_scale=1):
        """
        Init game for trading with 4 actions: Buy, Sell, Close Position, Hold
        
        Parameters
        ----------
        gym_env (gym_class): 
            The gym environment simulating the trading market.
            Defaults to None.
        
        discount (float): 
            The discount factor for the value calculation.
            Defaults to 0.95.
        
        limit_of_game_play (int): 
            Maximum number of game plays allowed per session.
            Defaults to float("inf").
        
        observation_dimension (int): 
            The dimension of the observation space.
            Defaults to None.
        
        action_dimension (int): 
            The dimension of the action space (set to 4 for Buy, Sell, Close, Hold).
            Defaults to 4.
        
        action_map (dict): 
            A dictionary mapping integers to the four actions.
            Defaults to None.
        
        priority_scale (float):
            Scaling the priority value.
            Defaults to 1.
        """     
        self.env = gym_env
        self.action_map = action_map if action_map else {0: 'Buy', 1: 'Sell', 2: 'Close', 3: 'Hold'}

        self.discount = discount
        self.limit_of_game_play = limit_of_game_play
        self.action_space_size = action_dimension
        self.priority_scale = priority_scale

        # Game storage
        self.action_history = []
        self.rewards = []
        self.policies = []
        self.root_values = []
        self.child_visits = []
        self.observations = []
        
        self.done = False

    def observation(self, observation_shape=None, iteration=0, feedback=None):
        """
        Handles getting the current observation state from the environment.
        If feedback is given, it uses the provided feedback.
        """
        if iteration == 0 and feedback is None:
            state = self.env.reset(seed=random.randint(0, 100000))[0]  # Grab state from reset
        elif feedback is not None:
            state = feedback.observations[iteration]
        else:
            state = self.feedback_state
        self.feedback_state = state
        return state

    def step(self, action):
        """
        Take an action and process the result from the environment.
        """
        next_step = self.env.step(action)
        return next_step

    def select_action(self, action, policy, temperature):
        """
        Select an action based on the policy and temperature value.
        """
        if temperature > 0.1 or len(set(policy)) == 1:
            selected_action = np.random.choice(action, p=policy)
        else:
            selected_action = action[np.argmax(policy)]
        return selected_action

    def policy_action_reward_from_tree(self, root):
        """
        Retrieve actions, policies, and rewards from a tree node (like an MCTS root).
        """
        action = np.array(list(root.children.keys()))
        policy = np.array([root.children[u].visit_count for u in list(root.children.keys())], dtype=np.float64)
        if policy.sum() <= 1:
            policy = np.array([root.children[u].prior for u in list(root.children.keys())], dtype=np.float64)
        reward = np.array([root.children[u].reward for u in list(root.children.keys())], dtype=np.float64)
        return action, policy, reward

    def onehot_action_encode(self, selected_action):
        """
        One-hot encode the selected action.
        """
        action_onehot_encoded = np.zeros(self.action_space_size)
        action_onehot_encoded[selected_action] = 1
        return action_onehot_encoded

    def policy_step(self, root=None, temperature=0, feedback=None, iteration=0):
        """
        Perform one step in the policy, given the root and temperature for exploration.
        """
        action, policy, reward = self.policy_action_reward_from_tree(root)
        policy = self.softmax_stable(policy, temperature=temperature)
        selected_action = self.select_action(action, policy, temperature)
        action_onehot_encoded = self.onehot_action_encode(selected_action)

        if isinstance(feedback, (tuple, type(None))):
            step_output = self.step(self.action_map[selected_action])

            # Direct state handling without RGB
            observation = step_output[0]

            step_val = (observation,) + step_output[1:]
        else:
            step_val = [feedback.observations[iteration+1],
                        feedback.rewards[selected_action+1],
                        iteration + 2 >= len(feedback.observations)-1]

        self.observations.append(step_val[0])
        self.rewards.append(step_val[1])
        self.policies.append(policy)
        self.action_history.append(action_onehot_encoded)

        c_max_limit = self.limit_of_game_play != len(self.observations)
        self.done = step_val[2] if c_max_limit else False

    def softmax_stable(self, policy, temperature=1.0):
        """
        Computes a numerically stable softmax for the policy.
        """
        z = policy - np.max(policy)
        exp = np.exp(z / temperature)
        return exp / np.sum(exp)
    
    def flatten_state(self, state):
        """
        Flatten the state to handle observation space appropriately.
        """
        return np.array(state).flatten()

    def render(self):
        """
        This method can be used if you want to render the environment (e.g., for debugging).
        """
        return self.env.render(mode='human')
