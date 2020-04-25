import torch.nn.functional as F

# Default hyperparameters
                           
SEED = 10                          # Random seed
NB_EPISODES = 10000                # Max nb of episodes
NB_STEPS = 1000                    # Max nb of steps per episodes 
UPDATE_EVERY_NB_EPISODE = 4        # Nb of episodes between learning process
MULTIPLE_LEARN_PER_UPDATE = 3      # Nb of multiple learning process performed in a row

BUFFER_SIZE = int(1e5)             # Replay buffer size
BATCH_SIZE = 256                   # Batch size #128
ACTOR_FC1_UNITS = 512   	       # Number of units for L1 in the actor model #256 
ACTOR_FC2_UNITS = 256         	   # Number of units for L2 in the actor model #128 
CRITIC_FCS1_UNITS = 512        	   # Number of units for L1 in the critic model #256 
CRITIC_FC2_UNITS = 256         	   # Number of units for L2 in the critic model #128 
NON_LIN = F.relu      			   # Non linearity operator used in the model #F.leaky_relu
LR_ACTOR = 1e-4              	   # Learning rate of the actor #1e-4 
LR_CRITIC = 5e-3             	   # Learning rate of the critic #1e-3 
WEIGHT_DECAY = 0            	   # L2 weight decay #0.0001

GAMMA = 0.995                 	   # Discount factor #0.99
TAU = 1e-3                         # For soft update of target parameters
CLIP_CRITIC_GRADIENT = False       # Clip gradient during Critic optimization

ADD_OU_NOISE = True     		   # Toggle Ornstein-Uhlenbeck noisy relaxation process
THETA = 0.15            		   # k/gamma -> spring constant/friction coefficient [Ornstein-Uhlenbeck]
MU = 0.                 		   # x_0 -> spring length at rest [Ornstein-Uhlenbeck]
SIGMA = 0.2             		   # root(2k_B*T/gamma) -> Stokes-Einstein for effective diffision [Ornstein-Uhlenbeck]
NOISE = 1.0                        # Initial Noise Amplitude 
NOISE_REDUCTION = 0.995 	       # Noise amplitude decay ratio
