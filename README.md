# This is a fork of Stochastic MuZero for the purpose of trimming down for application
docker run -it --cpus 2 --shm-size=2.36gb -v "G:/My Drive/ZBAIDA_LLC/Trading/smu0:/app" smu0:latest
# Explanation of workings of stochastic muzero algorithm:
The Stochastic MuZero (SMuZero) algorithm is an advanced model-based reinforcement learning (RL) method, extending the MuZero framework by incorporating stochastic environments and learning a latent model that captures both the dynamics of the environment and the agent’s policy. Here’s an overview of how it works and how the functional units interact:
Overview of Stochastic MuZero (SMuZero)

Stochastic MuZero is an offline reinforcement learning algorithm, meaning it doesn't rely on explicit environmental models (like MDPs) but instead learns both a model and a policy directly from data. It combines a deep neural network architecture that learns latent representations and a planning method based on Monte Carlo Tree Search (MCTS). Here’s a breakdown of the components:

    Representation Model:
        The representation model takes the current state and the action taken by the agent and produces a latent representation of the environment's state.

    Dynamics Model:
        The dynamics model predicts the next latent state and the reward, based on the current latent state and the action taken.

    Prediction Model:
        The prediction model outputs the expected value of the current latent state (i.e., the policy and the value), which helps decide the agent's next action during training and inference.

    Value Model:
        The value model is used to estimate the expected cumulative reward (value) of a state or action.

    Planning with MCTS:
        MCTS (Monte Carlo Tree Search) is used for planning. It simulates trajectories of states and actions, using the learned models to estimate future rewards and the best actions to take.

Functional Units and Their Interactions

Let’s break down the major functional units and how they communicate during both training and action-selection phases.
1. Experience Collection & Environment Interaction

    What Happens:
        The agent interacts with the environment by taking actions and receiving rewards.
        This interaction generates experience tuples: (state, action, reward, next_state).
        These experiences are typically stored in an experience replay buffer.

    When It Happens:
        This occurs during the agent’s interaction with the environment, typically in real-time or in a simulation.
        The experience is used for training later, providing a set of diverse examples to learn from.

2. Latent State Representation (Function: Representation Model)

    What Happens:
        The representation model takes the current state as input and outputs a latent representation. This model learns to map raw states (e.g., images, observations) into a more compact and useful representation for downstream tasks.
        The agent receives raw state inputs, which could be high-dimensional, and the model outputs a lower-dimensional latent state representation.

    When It Happens:
        This happens both during experience collection (to create a latent state) and during planning (when predicting the next state).

    Why It Happens:
        The representation allows the algorithm to learn a model that can predict future states from the current latent state, which is crucial for MCTS planning.

3. Dynamics Model (Function: Dynamics Model)

    What Happens:
        The dynamics model learns to predict how the environment evolves given a latent state and an action. It outputs two things:
            Next latent state: This is the latent state after taking the action in the current latent state.
            Reward: This is the expected reward after taking the action.
    When It Happens:
        During both training (learning the environment's dynamics) and planning (simulating future states in MCTS).
    Why It Happens:
        The dynamics model allows the algorithm to simulate what would happen if it took a particular action in a particular state. This is essential for the planning phase where the agent needs to predict future states and rewards.

4. Prediction Model (Function: Prediction Model)

    What Happens:
        The prediction model outputs a probability distribution over actions (policy) and the expected value of the state (value function) given a latent state.
    When It Happens:
        During training, this model helps calculate the loss for policy and value predictions.
        During planning (via MCTS), the prediction model estimates the reward and value of different state-action pairs.
    Why It Happens:
        The prediction model is key for guiding the agent’s behavior. It helps the agent decide what actions to take by estimating which ones are most likely to lead to higher rewards.

5. Monte Carlo Tree Search (MCTS) (Function: MCTS)

    What Happens:
        MCTS simulates the future by recursively applying the dynamics model and prediction model to explore possible actions.
        It performs simulation by creating a tree of possible future states. At each node, it uses the prediction model to select actions, and the dynamics model to simulate the next state.
        After running simulations, the algorithm backpropagates the estimated rewards to adjust the value of states in the tree.
    When It Happens:
        Planning phase: This is invoked every time the agent needs to make a decision about which action to take in a particular state.
    Why It Happens:
        MCTS allows the agent to simulate and evaluate the possible consequences of its actions before making a decision, improving its ability to select optimal actions, even in a stochastic environment.

6. Learning Phase (Function: Learning Loop)

    What Happens:
        The agent periodically updates its models (representation, dynamics, prediction) using the experiences collected and the predictions from MCTS.
        Loss Calculation: The algorithm computes the loss based on the difference between predicted rewards and true rewards, predicted values and actual values, and predicted policies and actual policies.
        Gradient Update: The loss is used to compute gradients and update the neural network weights.

    When It Happens:
        The learning phase happens periodically, after collecting enough experience, to update the model parameters.

    Why It Happens:
        The learning phase ensures that the models (representation, dynamics, and prediction) improve over time. This is necessary for the algorithm to generalize better to unseen states.

7. Action Selection (Function: Policy Selection via MCTS)

    What Happens:
        After planning via MCTS, the agent selects an action based on the simulations (i.e., the tree search).
        The action is chosen based on the estimated value or reward at the tree’s leaf nodes.

    When It Happens:
        Every time the agent needs to make a decision, whether during training or deployment.

    Why It Happens:
        Action selection through MCTS ensures that the agent chooses actions based on its learned understanding of the environment, maximizing expected cumulative rewards.

Flow of Operations:

    Collect Experience:
        The agent interacts with the environment and stores (state, action, reward, next_state) tuples.

    Represent States:
        The representation model is used to transform states into latent representations.

    Plan (MCTS):
        For each decision step, MCTS simulates future actions using the dynamics and prediction models to estimate rewards and values.

    Update Models:
        After collecting enough data and planning, the agent updates its models (representation, dynamics, prediction) using supervised learning and value optimization.

    Select Action:
        After MCTS planning, the agent selects the action that leads to the highest expected reward.

When Do Different Functional Units Communicate?

    Experience Buffer ↔ Representation Model:
        The experience buffer feeds raw state data into the representation model to create latent states for training.

    Representation Model ↔ Dynamics & Prediction Models:
        The representation model interacts with the dynamics and prediction models to provide latent state representations, which are used in simulations and reward predictions.

    Prediction & Dynamics Models ↔ MCTS:
        MCTS relies on the prediction model to evaluate states and actions, while it uses the dynamics model to simulate future states during planning.

    MCTS ↔ Action Selection:
        The result of the MCTS planning phase directly informs the action selection step, as the agent chooses the action with the best expected reward.

    Models ↔ Learning:
        The models (representation, dynamics, prediction) are periodically updated via backpropagation based on the discrepancy between predicted and actual rewards and values.

In summary, Stochastic MuZero combines planning (MCTS) and learning (neural network-based models) in a way that allows the agent to simulate potential futures and optimize its policy even in complex, stochastic environments. The communication between different units (experience buffer, representation, dynamics, prediction, MCTS, learning) ensures that the agent continually refines its internal models and makes decisions based on a combination of past experience and predicted outcomes.



Plan for Implementing Stochastic MuZero (SMuZero) in Python

To build a Stochastic MuZero (SMuZero) implementation, we need to organize the system into manageable components. Each component will encapsulate specific functionality, from interacting with the environment to learning and planning. Here’s an outline of the necessary components and how they will work together.
High-Level Overview

    Environment Interaction:
        The agent will interact with the environment by taking actions and receiving rewards.

    Latent Representation:
        A deep neural network will be used to learn latent representations of the state, which helps in planning and decision-making.

    Dynamics Model:
        This model will predict the next state and reward, given a current state and action.

    Prediction Model:
        This model will estimate the expected value and policy for the current latent state.

    Planning (MCTS):
        The Monte Carlo Tree Search (MCTS) will simulate future states using the learned models and select the best action.

    Learning:
        The algorithm will use the experiences to update the models (representation, dynamics, and prediction models) using supervised learning.

    Action Selection:
        The agent will select an action based on the planning results from MCTS.

Classes and Their Functions
1. Environment (Env)

    Purpose: To simulate the environment, return the next state and reward based on actions taken by the agent.
    Methods:
        reset(): Resets the environment to the initial state.
        step(action): Takes an action and returns (next_state, reward, done).

2. RepresentationModel

    Purpose: To encode the raw state into a compact latent state representation.
    Methods:
        forward(state): Takes the raw state as input and outputs a latent state.

3. DynamicsModel

    Purpose: To predict the next latent state and the reward given the current latent state and action.
    Methods:
        forward(state, action): Given a latent state and action, predicts the next latent state and reward.

4. PredictionModel

    Purpose: To output the value and policy (probabilities of each action) for a given latent state.
    Methods:
        forward(state): Outputs the policy and value for the given latent state.

5. MCTS

    Purpose: To simulate the environment and select the best action using Monte Carlo Tree Search (MCTS).
    Methods:
        search(root_state, model): Runs MCTS to find the best action from a root state using the provided model.
        expand(node): Expands a node by applying the model and simulating future states.

6. Agent

    Purpose: To tie everything together by interacting with the environment, learning from experience, and selecting actions.
    Methods:
        act(state): Selects an action based on the current state (using MCTS for planning).
        learn(): Updates the models using stored experiences and the learning procedure.

Plan for Implementation

    Environment Simulation:
        We will use a simple grid-world environment or any other discrete environment where the agent’s actions can be mapped easily to state transitions and rewards.
        We will simulate the environment’s stochasticity by introducing randomness in the state transitions and rewards.

    Neural Network Models:
        Use simple feedforward neural networks to model the representation, dynamics, and prediction models. We can use PyTorch or implement them from scratch using NumPy.

    MCTS Implementation:
        Implement Monte Carlo Tree Search for planning. The tree will have nodes representing latent states, and at each node, we will store the policy and value predicted by the prediction model.
        MCTS will simulate future states by applying the dynamics model and selecting actions based on the policy.

    Learning:
        The learning process will be based on the data collected during the agent’s interaction with the environment.
        We will use Supervised Learning for updating the models by comparing the model's predictions with actual rewards and states from the environment.

    Action Selection:
        MCTS will be used to choose the best action by simulating future trajectories from the current state.

# Pseudocode in Python
Classes and Functions

import numpy as np

class Environment:
    def __init__(self):
        # Initialize environment
        pass

    def reset(self):
        # Reset the environment and return initial state
        return np.zeros((5, 5))  # example state

    def step(self, action):
        # Simulate environment response to an action
        next_state = np.zeros((5, 5))  # example next state
        reward = np.random.random()  # example random reward
        done = False  # example done condition
        return next_state, reward, done


class RepresentationModel:
    def __init__(self):
        # Initialize a simple model (could be a neural net or a simple transformation)
        pass

    def forward(self, state):
        # Convert state to a latent state
        return np.random.rand(10)  # latent state representation


class DynamicsModel:
    def __init__(self):
        # Initialize the dynamics model (a simple neural network)
        pass

    def forward(self, state, action):
        # Predict next state and reward given current state and action
        next_state = np.random.rand(10)  # next latent state
        reward = np.random.random()  # reward prediction
        return next_state, reward


class PredictionModel:
    def __init__(self):
        # Initialize the prediction model
        pass

    def forward(self, state):
        # Predict value and policy for a given state
        value = np.random.random()  # value of the state
        policy = np.random.rand(4)  # action probabilities
        return value, policy


class MCTS:
    def __init__(self, model, env):
        self.model = model
        self.env = env

    def search(self, root_state):
        # Perform MCTS to determine the best action
        node = root_state
        for _ in range(100):  # Run 100 simulations
            self.expand(node)
        return np.argmax(node['policy'])  # Return the action with the highest policy

    def expand(self, node):
        # Simulate future actions using the dynamics model
        action = np.random.choice(4)  # Example: 4 possible actions
        next_state, reward = self.model.forward(node, action)
        node['value'], node['policy'] = self.model.forward(next_state)
        return node


class Agent:
    def __init__(self, env):
        self.env = env
        self.rep_model = RepresentationModel()
        self.dyn_model = DynamicsModel()
        self.pred_model = PredictionModel()
        self.mcts = MCTS(self.pred_model, env)

    def act(self, state):
        # Select action using MCTS
        latent_state = self.rep_model.forward(state)
        action = self.mcts.search(latent_state)
        return action

    def learn(self, experiences):
        # Learn from experiences by updating models (e.g., using supervised learning)
        pass

High-Level Flow

    Initialization:
        The environment and agent are initialized.
        The agent's models (representation, dynamics, prediction) are also initialized.

    Training Loop:
        The agent interacts with the environment by selecting actions using the act() method.
        The environment returns the next state and reward.
        The agent stores the experience (state, action, reward, next_state).

    Planning (MCTS):
        For each action selection, MCTS is performed by the agent to simulate the best actions using the learned models.

    Learning:
        After collecting enough experience, the agent updates its models using supervised learning or some other suitable optimization method.

# Conclusion

This plan provides a clear structure for implementing Stochastic MuZero using a simple Python setup without relying on large, complex libraries. Each component (environment, models, MCTS, and agent) is modular, allowing for flexibility and easier debugging. The pseudocode outlines the basic interaction between these components and serves as a foundation for further development of the algorithm.



Pytorch Implementation of [Stochastic MuZero](https://openreview.net/pdf?id=X6D9bAHhBQ1). Base on [Muzero Unplugged](https://github.com/DHDev0/Muzero-unplugged). 

It is suggested to refer to Stochastic MuZero as "unplugged," as setting the reanalyze_ratio to 0 is necessary to achieve Stochastic MuZero. This is because the original "Stochastic MuZero" paper highlights online reinforcement learning, however, as an enhancement to "MuZero Unplugged," it also encompasses offline reinforcement learning capabilities.

[MuZero](https://github.com/DHDev0/Muzero) -> [MuZero Unplugged](https://github.com/DHDev0/Muzero-unplugged) -> [Stochastic MuZero](https://github.com/DHDev0/Stochastic-muzero)

*A scheduled update is planned for the release of PyTorch 2.1.

Table of contents
=================
<!--ts-->
   * [Getting started](#getting-started)
      * [Local Installation](#local-installation)
      * [Docker](#docker)
      * [Dependency](#dependency)
   * [Usage](#usage)
      * [Jupyter Notebook](#jupyter-notebook)
      * [CLI](#cli)
   * [Features](#features)
   * [How to make your own custom gym environment?](#how-to-make-your-own-custom-gym-environment)
   * [Authors](#authors)
   * [Subjects](#subjects)
   * [License](#license)

<!--te-->

Getting started
===============

Local Installation
------------------

PIP dependency : [requirement.txt](https://github.com/DHDev0/Stochastic-muzero/blob/main/requirements.txt)
~~~bash
git clone https://github.com/DHDev0/Stochastic-muzero.git

cd Stochastic-muzero

pip install -r requirements.txt
~~~

If you experience some difficulty refer to the first cell [Tutorial](https://github.com/DHDev0/Stochastic-muzero/blob/main/tutorial.ipynb) or use the dockerfile.

Docker
------
 
Build image: (building time: 22 min , memory consumption: 8.75 GB)
~~~bash
docker build -t stochastic_muzero .
~~~ 
(do not forget the ending dot)

Start container:
~~~bash
docker run --cpus 2 --gpus 1 -p 8888:8888 stochastic_muzero
#or
docker run --cpus 2 --gpus 1 --memory 2000M -p 8888:8888 stochastic_muzero
#or
docker run --cpus 2 --gpus 1 --memory 2000M -p 8888:8888 --storage-opt size=15g stochastic_muzero
~~~ 

The docker run will start a jupyter lab on https://localhost:8888//lab?token=token (you need the token) with all the necessary dependency for cpu and gpu(Nvidia) compute.

Option meaning:  
--cpus 2 -> Number of allocated (2) cpu core  
--gpus 1 -> Number of allocated (1) gpu  
--storage-opt size=15gb -> Allocated storage capacity 15gb (not working with windows WSL)  
--memory 2000M -> Allocated RAM capacity of 2GB  
-p 8888:8888 -> open port 8888 for jupyter lab (default port of the Dockerfile)  

Stop the container:
~~~bash
docker stop $(docker ps -q --filter ancestor=stochastic_muzero)
~~~ 

Delete the container:
~~~bash
docker rmi -f stochastic_muzero
~~~ 

Dependency
----------
Language : 
* Python 3.8 to 3.10
(bound by the retro compatibility of Ray and Pytorch)

Library : 
* torch 1.13.0
* torchvision 0.14.0
* ray 2.0.1 
* gymnasium 0.27.0
* matplotlib >=3.0
* numpy 1.21.5

More details at: [requirement.txt](https://github.com/DHDev0/Stochastic-muzero/blob/main/requirements.txt)


Usage
=====

Jupyter Notebook
---------------

For practical example, you can use the [Tutorial](https://github.com/DHDev0/Stochastic-muzero/blob/main/tutorial.ipynb).


CLI
-----------

Set your config file (example): https://github.com/DHDev0/Stochastic-muzero/blob/main/config/

First and foremost cd to the project folder:
~~~bash 
cd Stochastic-muzero
~~~

Construct your dataset through experimentation.
~~~bash 
python muzero_cli.py human_buffer config/experiment_450_config.json
~~~

Training :
~~~bash 
python muzero_cli.py train config/experiment_450_config.json
~~~  

Training with report
~~~bash
python muzero_cli.py train report config/experiment_450_config.json
~~~  

Inference (play game with specific model) :
~~~bash 
python muzero_cli.py train play config/experiment_450_config.json
~~~ 

Training and Inference :
~~~bash 
python muzero_cli.py train play config/experiment_450_config.json
~~~  

Benchmark model :
~~~bash
python muzero_cli.py benchmark config/experiment_450_config.json
~~~ 

Training + Report + Inference + Benchmark :
~~~python 
python muzero_cli.py train report play benchmark play config/experiment_450_config.json
~~~  

Features
========

Core Muzero and Muzero Unplugged features:
* [x] Work for any Gymnasium environments/games. (any combination of continous or/and discrete action and observation space)
* [x] MLP network for game state observation. (Multilayer perceptron)
* [x] LSTM network for game state observation. (LSTM)
* [x] Transformer decoder for game state observation. (Transformer)
* [x] Residual network for RGB observation using render. (Resnet-v2 + MLP)
* [x] Residual LSTM network for RGB observation using render. (Resnet-v2 + LSTM)
* [x] MCTS with 0 simulation (use of prior) or any number of simulation.
* [x] Model weights automatically saved at best selfplay average reward.
* [x] Priority or Uniform for sampling in replay buffer.
* [X] Manage illegal move with negative reward.
* [X] Scale the loss using the importance sampling ratio.
* [x] Custom "Loss function" class to apply transformation and loss on label/prediction.
* [X] Load your pretrained model from tag number.
* [x] Single player mode.
* [x] Training / Inference report. (not live, end of training)
* [x] Single/Multi GPU or Single/Multi CPU for inference, training and self-play.
* [x] Support mix precision for training and inference.(torch_type: bfloat16,float16,float32,float64)
* [X] Pytorch gradient scaler for mix precision in training.
* [x] Tutorial with jupyter notebook.
* [x] Pretrained weights for cartpole. (you will find weight, report and config file)
* [x] Commented with link/page to the paper.
* [x] Support : Windows , Linux , MacOS.
* [X] Fix pytorch linear layer initialization. (refer to : https://tinyurl.com/ykrmcnce)
* [X] Support of [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 0.27.0  
* [X] The ability to accommodate any number of players with the provision of player cycle information.
* [X] The incorporation of reanalyze buffer(offline learning) and reanalyze ratio functionality.
* [X] The capability to construct human play datasets through experimentation (CLI only).
* [X] The capability to load human play datasets into the Demonstration buffer or Replay buffer for training.
* [X] The ability to specify the amount of sampled action that MCTS should utilize.
* [X] The implementation of a priority scale on neural network and replay buffer priority.
* [X] Various options for bounding, saving, and deleting games from the reanalyze buffer.
* [X] The introduction of the reanalyze_fraction_mode, which allows for the statistical or  
quantitative switch between new game and reanalyze data with a ratio of reanalyze buffer vs replay buffer."
* [X] The implementation of a scaling parameter of the value loss.

Muzero Stochastic new add-on features include: 
* [X] No gradient scaling.
* [X] Add model of afterstate_prediction_function, aftstate_dynamic_function and encoder_function.
* [X] Extend batch with all observation following an initial index.
* [X] Extend mcts with chance node.
* [X] Extend forward pass with afterstate_prediction, aftstate_dynamic and encoder.
* [X] Extend loss function with value_afterstate_loss, distribution_afterstate_loss and vq-vae_commitment_cost.
* [X] [Encoder] The encoder embedding c_e_t is modeled as a categorical variable.
* [X] [Encoder] Selecting the closest code c_t is equivalent to computing the expression one_hot(arg_max(c_e_t)).
* [X] [Encoder] Use of the Gumbel-Softmax reparameterization trick with zero temperature during the forward pass. (meaning you just forward the encoder with random noise during training and without noise during inference. Since the temperature is 0 you don't forward anything)
* [X] [Encoder] A straight-through estimator is used during the backward of the encoder to allow the gradients to flow only to the encoder during the backpropagation.
* [X] [Encoder] There is no explicit decoder in the model and it does not use a reconstruction loss.
* [X] [Encoder] The network is trained end-to-end in a fashion similar to MuZero.

TODO:
* [ ] Hyperparameter search. (pseudo-code available in self_play.py)
* [ ] Training and deploy on cloud cluster using Kubeflow, Airflow or Ray for aws,gcp and azure.



How to make your own custom gym environment?
================================================

Refer to the [Gym documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)

You will be able to call your custom gym environment in muzero after you register it in gym.

Authors  
==========

- [Daniel Derycke](https://github.com/DHDev0)  

Subjects
========

Deep reinforcement learning


License 
=======

[GPL-3.0 license](https://www.gnu.org/licenses/quick-guide-gplv3.html)  



