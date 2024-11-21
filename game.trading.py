# chatgpt modified game.py for trading game

'''
This code defines a Game class for interacting with a gym environment and implementing some form of Monte Carlo Tree Search (MCTS), potentially for MuZero-style reinforcement learning, based on the provided pseudocode.
Key Features of the Game Class:

    Environment Setup:
        Initializes a gym environment with observation dimensions, action space size, and optional RGB observations.
        Handles discount factors and maximum steps per game session.

    Observation Handling:
        Processes the environment's observations, either as flattened state arrays or transformed RGB images (if the observation is an image).
        Manages the reanalyzing of states and resets for new games.

    Action Selection:
        Uses softmax to generate a probability distribution over actions from the MCTS search tree's visit counts.
        Selects actions based on this policy, either through stochastic selection or greedily picking the action with the highest probability.

    Rewards and Game Progression:
        Tracks rewards, actions, policies, and other key gameplay data such as MCTS root values and child visits.
        Implements functionality to step through the environment using selected actions and processes results from environment feedback.

    Prioritization:
        Computes priority values for different steps of the game based on how far the estimated root value of the game diverges from the target value after several time steps (td_steps).

    Key Mapping:
        Allows the user to define a custom keyboard mapping for controlling the environment during gameplay.

Important Methods:

    step(): Takes an action and processes the result in the gym environment.
    observation(): Handles getting the current observation state, either directly from the environment or from feedback data.
    select_action(): Chooses an action from the available options based on the policy.
    store_search_statistics(): Stores policy data from the MCTS for future use.
    make_target(): Creates targets for value and policy learning by considering rewards and future root values.
    play_record(): A method for live gameplay where the user controls the game with keyboard inputs, which can be recorded.
'''

import numpy as np
import torchvision.transforms as transforms
import torch
import random
import gymnasium as gym
import json

class Game():
    def __init__(self, 
                 gym_env=None, discount=0.95, limit_of_game_play=float("inf"), 
                 observation_dimension=None, action_dimension=4, 
                 rgb_observation=None, action_map=None, priority_scale=1):
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
        
        rgb_observation (bool): 
            Whether to use RGB render as the observation.
            Defaults to None.
        
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
        assert isinstance(discount, float) and discount >= 0, "Discount should be a positive float"
        self.limit_of_game_play = limit_of_game_play
        assert isinstance(limit_of_game_play, (float, int)) and limit_of_game_play >= 0, "Limit of game play should be a positive integer or float"
        self.action_space_size = action_dimension
        assert isinstance(action_dimension, int) and action_dimension >= 1, "Action dimension should be a positive integer"
        self.rgb_observation = rgb_observation
        assert isinstance(rgb_observation, bool), "RGB observation should be a boolean"
        self.done = False
        assert isinstance(self.done, bool), "Done should be a boolean"
        self.priority_scale = priority_scale
        assert isinstance(priority_scale, (float, int)) and 0 <= priority_scale <= 1, "Priority scale should be between 0 and 1"

        # Game storage
        self.action_history = []
        self.rewards = []
        self.policies = []
        self.root_values = []
        self.child_visits = []
        self.observations = []
        
        self.reanalyzed = False

        shape = observation_dimension[:-1] if isinstance(observation_dimension, tuple) else None
        if shape:
            self.transform_rgb = transforms.Compose([
                lambda x: x.copy().astype(np.uint8),
                transforms.ToTensor(),
                transforms.Resize(shape),
                lambda x: x[None, ...]  # Add batch dimension
            ])
        else:
            self.transform_rgb = None

    def observation(self, observation_shape=None, iteration=0, feedback=None):
        if iteration == 0 and feedback is None: 
            state = self.env.reset(seed=random.randint(0, 100000))
            if self.rgb_observation:
                try:
                    state = self.tuple_test_obs(self.render())
                except:
                    state = self.transform_rgb(self.tuple_test_obs(state))
            else:
                state = self.flatten_state(self.tuple_test_obs(state))
        elif not isinstance(feedback, (tuple, type(None))):
            state = feedback.observations[iteration]
            if iteration == 0:
                self.reanalyzed = True
        else:
            state = feedback[0]
        self.feedback_state = state
        return state

    def step(self, action):
        try: 
            next_step = self.env.step(action)
        except:
            obs = self.feedback_state
            reward = min(-len(self.rewards), -self.limit_of_game_play, -1)
            done = self.done
            next_step = (obs, reward, done)
        return next_step

    def select_action(self, action, policy, temperature):
        if temperature > 0.1 or len(set(policy)) == 1:
            selected_action = np.random.choice(action, p=policy)
        else:
            selected_action = action[np.argmax(policy)]
        return selected_action

    def policy_action_reward_from_tree(self, root):
        action = np.array(list(root.children.keys()))
        policy = np.array([root.children[u].visit_count for u in list(root.children.keys())], dtype=np.float64)
        if policy.sum() <= 1:
            policy = np.array([root.children[u].prior for u in list(root.children.keys())], dtype=np.float64)
        reward = np.array([root.children[u].reward for u in list(root.children.keys())], dtype=np.float64)
        return action, policy, reward

    def onehot_action_encode(self, selected_action):
        action_onehot_encoded = np.zeros(self.action_space_size)
        action_onehot_encoded[selected_action] = 1
        return action_onehot_encoded

    def policy_step(self, root=None, temperature=0, feedback=None, iteration=0):
        action, policy, reward = self.policy_action_reward_from_tree(root)
        policy = self.softmax_stable(policy, temperature=temperature)
        selected_action = self.select_action(action, policy, temperature)
        action_onehot_encoded = self.onehot_action_encode(selected_action)

        if isinstance(feedback, (tuple, type(None))):
            step_output = self.step(self.action_map[selected_action])

            if self.rgb_observation: 
                try: observation = self.render()
                except: observation = self.transform_rgb(step_output[0])
            else:
                observation = self.flatten_state(step_output[0])

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
        
        Parameters:
        -----------
        policy : np.array
            The input array of action probabilities.
        temperature : float
            The scaling factor for exploration. Defaults to 1.0.
        
        Returns:
        --------
        np.array
            The softmax distribution.
        """
        z = policy - np.max(policy)
        exp = np.exp(z / temperature)
        return exp / np.sum(exp)
    
    def flatten_state(self, state):
        """
        Flatten the state to handle observation space appropriately.
        """
        return np.array(state).flatten()

    def tuple_test_obs(self, state):
        """
        Helper function to convert tuple state to suitable observation format.
        """
        return np.asarray(state) if isinstance(state, tuple) else state

    def render(self):
        """
        Returns rendered environment, useful for RGB observations.
        """
        return self.env.render(mode='rgb_array')
