# # # # # # # # # # # # # # # # # # # # # # # # # # Install dependencies # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Python 3.8.3 to 3.9.0 to 3.10 (3.10 is experimental with ray)
# Uncomment to install dependency
# !pip3 install matplotlib
# !pip3 install numpy

# Uncomment to install dependency
# For Nvidia GPU/CPU version (didn't try with ROCm for AMD GPU)
# update nvidia driver
# download and install cuda from: https://developer.nvidia.com/cuda-11-6-1-download-archive
# install cudnn: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
# !pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Uncomment to install dependency
# For CPU only
# The model is fairly small for CartPole, so it should run well on CPU only
# !pip3 install torch torchvision torchaudio

# Uncomment to install dependency
# Gym env for game simulation
# Install for Windows: Microsoft C++ Build Tools 14+ (required for Box2D lib)
# https://visualstudio.microsoft.com/downloads/
# !pip3 install gym[all]
# !pip3 install gym[atari]
# !pip install gym[accept-rom-license]
# or
# conda install -c conda-forge gym-all
# If you can't install Box2D, try:
# For Windows: conda install -c anaconda swig
# or else try the docker.

# Ray install
# pip install -U "ray[default]"

# #### OR #####
# Replace "!pip3" by "!pip" depending on the OS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# For cloud compute:
# https://docs.ray.io/en/latest/cluster/vms/getting-started.html#vm-cluster-quick-start
# https://docs.ray.io/en/latest/train/train.html
# - Create Ray Kubernetes cluster at your cloud provider. (with or without GPU)
# - Add Kubernetes cluster address to ray.init() of self_play in self_play.py for remote cluster compute.
# - Wrap model muzero_model.train() in self_play.py into a ray.init() and add Kubernetes cluster address.
# - Use function or CLI as usual.

import gymnasium as gym
from monte_carlo_tree_search import *
from game import *
from replay_buffer import *
from self_play import *
from muzero_model import *

# Uncomment to print complete tensor
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

# Example for initializing and setting up the game environment
# Uncomment to use the game environment and see available structure
# Show_all_gym_env()
# Show_specific_gym_env_structure("CartPole-v1")

# Set the game environment
env = gym.make("CartPole-v1", render_mode=None)

# Set the random seed for reproducibility
seed = 0
np.random.seed(seed)  # Set the random seed of numpy
torch.manual_seed(seed)  # Set the random seed of PyTorch

# Initialize the MuZero model for training and inference
muzero = Muzero(
    model_structure='mlp_model',  # 'vision_model' uses RGB as observation, 'mlp_model' uses game state
    observation_space_dimensions=env.observation_space,  # Dimension of the observation
    action_space_dimensions=env.action_space,  # Dimension of the action (gym box/discrete)
    state_space_dimensions=61,  # Support size/encoding space
    hidden_layer_dimensions=126,  # Number of weights in the recursive layer of the MLP
    number_of_hidden_layer=4,  # Number of recursion layers of hidden layers in MLP
    k_hypothetical_steps=10,  # Number of future steps to simulate during training
    learning_rate=0.01,  # Learning rate
    optimizer="adam",  # Optimizer ('adam' or 'sgd')
    lr_scheduler="cosineannealinglr",  # Learning rate scheduler
    loss_type="general",  # Loss type ('general' or 'game')
    num_of_epoch=1000,  # Number of steps for lr_scheduler
    device="cpu",  # Hardware on which to compute ("cpu", "cuda")
    type_format=torch.float32,  # Data type of the model
    load=False,  # Function to load a saved model
    use_amp=False,  # Use mixed precision
    bin_method="uniform_bin",  # Action binning method ("linear_bin" or "uniform_bin")
    bin_decomposition_number=10,  # Number of actions to sample from the low/high bound of a gym discret box
    priority_scale=0.5,
    rescale_value_loss=1
)

# Initialize demonstration buffer
demonstration_buffer = DemonstrationBuffer()

# Initialize the game storage (store each game) and dataset (create dataset)
replay_buffer = ReplayBuffer(
    window_size=500,  # Number of games stored in the buffer
    batch_size=128,  # Batch size of observed games during training
    num_unroll=muzero.k_hypothetical_steps,  # Number of moves played inside the batched game
    td_steps=50,  # Number of steps for TD scaling
    game_sampling="priority",  # Sampling mode ('uniform' or 'priority')
    position_sampling="priority",  # Sampling position mode ('uniform' or 'priority')
    reanalyze_stack=[
        ReanalyseBuffer(),
        demonstration_buffer,
        MostRecentBuffer(max_buffer_size=10),
        HighestRewardBuffer(max_buffer_size=10)
    ],
    reanalyse_fraction=0.1,  # Fraction of reanalysis
    reanalyse_fraction_mode="chance"  # Reanalysis mode ("chance" or "ratio")
)

# Initialize Monte Carlo Tree Search (MCTS) parameters
mcts = Monte_carlo_tree_search(
    num_simulations=5,  # Number of tree levels 50 is slow? on cpu?
    maxium_action_sample=2,  # Number of nodes per level
    number_of_player=1,  # Cycles through each player
    pb_c_base=19652,
    pb_c_init=1.25,
    discount=0.997,
    root_dirichlet_alpha=0.25,
    root_exploration_fraction=0.25,
    custom_loop=None
)

# Initialize Game class
gameplay = Game(
    limit_of_game_play=500,  # Max number of moves
    gym_env=env,
    discount=mcts.discount,
    observation_dimension=muzero.observation_dimension,
    action_dimension=muzero.action_dimension,
    rgb_observation=muzero.is_RGB,
    action_map=muzero.action_dictionnary,
    priority_scale=muzero.priority_scale
)

# Print the dimensions of observation and action spaces
print(f"Dimension of the observation space: {muzero.observation_dimension}\n"
      f"Dimension of the action space: {muzero.action_dimension}")

# Train the model
epoch_pr, loss, reward, learning_config = learning_cycle(
    number_of_iteration=1000,
    number_of_self_play_before_training=10,
    number_of_training_before_self_play=1,
    model_tag_number=450,
    temperature_type="static_temperature",
    verbose=True,
    number_of_worker_selfplay=4,
    muzero_model=muzero,
    gameplay=gameplay,
    monte_carlo_tree_search=mcts,
    replay_buffer=replay_buffer
)

# Report and generate config file
from self_play import report, generate_config_file

report(muzero, replay_buffer, epoch_pr, loss, reward, verbose=True)
generate_config_file(env, seed, muzero, replay_buffer, mcts, gameplay, learning_config)

# Play game from checkpoint
from self_play import play_game_from_checkpoint
import torch

play_game_from_checkpoint(
    game_to_play='CartPole-v1',
    model_tag=450,
    model_device="cpu",
    model_type=torch.float32,
    mcts_pb_c_base=19652,
    mcts_pb_c_init=1.25,
    mcts_discount=0.997,
    mcts_root_dirichlet_alpha=0.25,
    mcts_root_exploration_fraction=0.25,
    mcts_with_or_without_dirichlet_noise=True,
    number_of_monte_carlo_tree_search_simulation=0,
    maxium_action_sample=2,
    number_of_player=1,
    custom_loop=None,
    temperature=0,
    game_iter=2000,
    slow_mo_in_second=0.0,
    render=True,
    verbose=True
)

# Benchmark
from self_play import benchmark

number_of_trial = 100
cache_t, cache_r, cache_a, cache_p = [], [], [], []

for _ in range(number_of_trial):
    tag, reward, action, policy = play_game_from_checkpoint(
        game_to_play='CartPole-v1',
        model_tag=450,
        model_device="cpu",
        model_type=torch.float32,
        mcts_pb_c_base=19652,
        mcts_pb_c_init=1.25,
        mcts_discount=0.997,
        mcts_root_dirichlet_alpha=0.25,
        mcts_root_exploration_fraction=0.25,
        mcts_with_or_without_dirichlet_noise=True,
        number_of_monte_carlo_tree_search_simulation=0,
        maxium_action_sample=2,
        number_of_player=1,
        custom_loop=None,
        temperature=0,
        game_iter=2000,
        slow_mo_in_second=0.0,
        render=False,
        verbose=False
    )

    cache_t.append(tag)
    cache_r.append(reward)
    cache_a.append(action)
    cache_p.append(policy)

benchmark(cache_r, cache_a, cache_t, cache_p)
