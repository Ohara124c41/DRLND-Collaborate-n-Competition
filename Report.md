# Deep Reinforcement Learning (MADDDPG) Agent Collaborate and Competition Project

### Introduction



[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Tennis"

[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


For this project, the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment was used.

![Trained Agent][image1]

## Environment Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.


The environment is considered solved, **when the average (over 100 episodes) of those scores is at least +0.5**.


## Agent Implementation

### Mulit-Agent Deep Deterministic Policy Gradient (MADDDPG)

[Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](https://arxiv.org/abs/1706.02275) builds upon the [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) algorithm where multiple agents are coordinating to complete tasks with only local information. In the viewpoint of one agent, the environment is non-stationary as policies of other agents are quickly upgraded and remain unknown. MADDPG is an actor-critic model redesigned particularly for handling such a changing environment and interactions between agents.


More concretely, consider a game with N agents with policies parameterized by θ = {θ<sub>1</sub>, ..., θ<sub>N</sub> }, and let π = {π<sub>1</sub>, ..., π<sub>N</sub> } be the set of all agent policies. Then we can write the gradient of the expected return for agent *i, J(θ<sub>i</sub>) = E[Ri]* as:

![image](https://github.com/Ohara124c41/DRLND-Collaborate-n-Competition/blob/master/images/formulae01.PNG?raw=true)


Here Q<sup>i</sup><sub>π</sub>(x, a<sub>1</sub>, ..., a<sub>N</sub> ) is a centralized action-value function that takes as input the actions of all
agents, a<sub>1</sub>, . . . , a<sub>N</sub> , in addition to some state information x, and outputs the Q-value for agent *i*. In
the simplest case, x could consist of the observations of all agents, x = (o<sub>1</sub>, ..., o<sub>N</sub> ), however we
could also include additional state information if available. Since each Q<sup>i</sup><sub>π</sub>
is learned separately, agents can have arbitrary reward structures, including conflicting rewards in a competitive setting.

We can extend the above idea to work with deterministic policies. If we now consider *N* continuous policies µθ<sub>i</sub> w.r.t. parameters θ<sub>i</sub> (abbreviated as µ<sub>i</sub>), the gradient can be written as:
∇θ<sub>i</sub>J(µ<sub>i</sub>) = E<sub>x,a</sub>∼D[∇θ<sub>i</sub>µ<sub>i</sub>(a<sub>i</sub>|o<sub>i</sub>)∇a<sub>i</sub>Q
µ
i
(x, a<sub>1</sub>, ..., a<sub>N</sub> )|a<sub>i</sub>=µ<sub>i</sub>(o<sub>i</sub>)],



The problem can be formalized in the multi-agent version of MDP, also known as Markov games. Say, there are N agents in total with a set of states S. Each agent owns a set of possible action, *A<sub>1</sub>,…,A<sub>N</sub>*, and a set of observation, O<sub>1</sub>,…,O<sub>N</sub>. The state transition function involves all states, action and observation spaces *T : S×A<sub>1</sub>×…A<sub>N</sub> ↦ S*. Each agent’s stochastic policy only involves its own state and action: πθ<sub>i</sub> : O<sub>i</sub>×A<sub>i</sub> ↦ [0,1], a probability distribution over actions given its own observation, or a deterministic policy: μθ<sub>i</sub> : O<sub>i</sub> ↦ A<sub>i</sub>.

Let <span style="text-decoration:overline">o</span> =o<sub>1</sub>,…,o<sub>N</sub>, <span style="text-decoration:overline">μ</span> =μ<sub>1</sub>,…,μ<sub>N</sub> and the policies are parameterized by <span style="text-decoration:overline">θ</span> =θ<sub>1</sub>,…,θ<sub>N</sub>.

The critic in MADDPG learns a centralized action-value function Q<sub>i</sub><sup><span style="text-decoration:overline">μ</span></sup>(<span style="text-decoration:overline">o</span> ,a<sub>1</sub>,…,a<sub>N</sub>) for the i-th agent, where a<sub>1</sub>∈A<sub>1</sub>,…,a<sub>N</sub>∈A<sub>N</sub> are actions of all agents. Each Q<sub>i</sub><sup><span style="text-decoration:overline">μ</span></sup> i is learned separately for i=1,…,N and therefore multiple agents can have arbitrary reward structures, including conflicting rewards in a competitive setting. Meanwhile, multiple actors, one for each agent, are exploring and upgrading the policy parameters θ<sub>1</sub> on their own.

Actor update:

![image](https://github.com/Ohara124c41/DRLND-Collaborate-n-Competition/blob/master/images/formulae02.PNG?raw=true)

Where D is the memory buffer for experience replay, containing multiple episode samples (<span style="text-decoration:overline">o</span>  , a<sub>1</sub>,…,a <sub>N</sub>, r<sub>1</sub>,…,r<sub>N</sub>, <span style="text-decoration:overline">o′</span> ) — given current observation <span style="text-decoration:overline">o</span> , agents take action a<sub>1</sub>,…,a<sub>N</sub> and get rewards r<sub>1</sub>,…,r<sub>N</sub>, leading to the new observation o  ′.

Critic update:

![image](https://github.com/Ohara124c41/DRLND-Collaborate-n-Competition/blob/master/images/formulae03.PNG?raw=true)

where <span style="text-decoration:overline">μ′</span> are the target policies with delayed softly-updated parameters.

If the policies <span style="text-decoration:overline">μ</span>  are unknown during the critic update, we can ask each agent to learn and evolve its own approximation of others’ policies. Using the approximated policies, MADDPG still can learn efficiently although the inferred policies might not be accurate.

To mitigate the high variance triggered by the interaction between competing or collaborating agents in the environment, MADDPG proposed one more element - policy ensembles:

- Train K policies for one agent;
- Pick a random policy for episode rollouts;
- Take an ensemble of these K policies to do gradient update.

In summary, MADDPG added three additional ingredients on top of DDPG to make it adapt to the multi-agent environment:

- Centralized critic + decentralized actors;
- Actors are able to use estimated policies of other agents for learning;
- Policy ensembling is good for reducing variance.


Below, the pseudocode is described:

#### Pseudocode
----------
![image](https://github.com/Ohara124c41/DRLND-Collaborate-n-Competition/blob/master/images/maddpg_pseudocode.PNG?raw=true)

## Model
The next two entries visualize the flow diagrams for the Network. This work builds upon implementations from the first DRLND project [Navigation](https://github.com/Ohara124c41/DRLND-Navigation/blob/master/Report.md) and the second DRLND project [Continuous Control](https://github.com/Ohara124c41/DRL-Continuous_Control/blob/master/Report.md).


#### Actor
Below, the flow diagram demonstrates how the Actor network is setup.

```py
model = Actor(state_size, action_size, 42)
model.eval()
x = Variable(torch.randn(1,state_size))
y = model(x)

make_dot(y, params=dict(list(model.named_parameters())))
```

![image](https://github.com/Ohara124c41/DRLND-Collaborate-n-Competition/blob/master/images/actor.PNG?raw=true)


#### Critic
Below, the flow diagram demonstrates how the Critic network is setup.

```py
model = Critic(state_size, action_size, 42)
model.eval()
x = Variable(torch.randn(1,state_size))
z = Variable(torch.randn(1,action_size))
y = model(x, z)

make_dot(y, params=dict(list(model.named_parameters())))
```

![image](https://github.com/Ohara124c41/DRLND-Collaborate-n-Competition/blob/master/images/critic.PNG?raw=true)



### Code Implementation
Note: the original OpenAI MADDPG implementation for policy ensemble and policy estimation can be found [here](https://www.dropbox.com/s/jlc6dtxo580lpl2/maddpg_ensemble_and_approx_code.zip?dl=0). The code is provided as-is.



**NB:** Code will run in GPU if CUDA is available, otherwise it will run in CPU.

Code is structured in different modules. The most relevant features will be explained next:

1. **model.py:** It contains the main execution thread of the program. This file is where the main algorithm is coded (see *algorithm* above). PyTorch is utilized for training the agent in the environment. The agent has an Actor and Critic network.
2. **ddpg_agent.py:** The model script contains  the **DDPG agent**, a **Replay Buffer memory**, and the **Q-Targets** feature.
3. **maddpg_agent.py:** Augments the DDPG `learn()` function for MADDPG via a `maddpg_learn()` method using batches to handle the value parameters and update the policy for `Q_targets`.
4. **hyperparameters.py:** Allows for a common module to inherit preset hyperparameters for all modules.
5. **memory.py:** Instantiates the buffer replay memory module.
6. **Tennis.ipynb:** The Navigation Jupyter Notebook provides an environment to run the *Tennis* game, import dependencies, train the MADDPG, visualize via Unity, and plot the results.


#### PyTorch Specifics

Saved Model Contents: PyTorch Version
-------------------------------------

The PyTorch saved model can be loaded with ``ac = torch.load('path/to/model.pt')``, yielding an actor-critic object (``ac``) that has the properties described in the docstring for ``ddpg_pytorch``.

You can get actions from this model with:


```py
actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))
```


### DDPG Hyperparameters

The DDPG agent uses the following parameters values (defined in parameters.py)

```
SEED = 10                           # Random seed
NB_EPISODES = 10000                 # Max nb of episodes
NB_STEPS = 1000                     # Max nb of steps per episodes
UPDATE_EVERY_NB_EPISODE = 4         # Nb of episodes between learning process
MULTIPLE_LEARN_PER_UPDATE = 3       # Nb of multiple learning process performed in a row

BUFFER_SIZE = int(1e5)              # Replay buffer size
BATCH_SIZE = 256                    # Batch size #128
ACTOR_FC1_UNITS = 512               # Number of units for L1 in the actor model #256
ACTOR_FC2_UNITS = 256               # Number of units for L2 in the actor model #128
CRITIC_FCS1_UNITS = 512             # Number of units for L1 in the critic model #256
CRITIC_FC2_UNITS = 256              # Number of units for L2 in the critic model #128
NON_LIN = F.relu                    # Non linearity operator used in the model #F.leaky_relu
LR_ACTOR = 1e-4                     # Learning rate of the actor #1e-4
LR_CRITIC = 5e-3                    # Learning rate of the critic #1e-3
WEIGHT_DECAY = 0                    # L2 weight decay #0.0001

GAMMA = 0.995                       # Discount factor #0.99
TAU = 1e-3                          # For soft update of target parameters
CLIP_CRITIC_GRADIENT = False        # Clip gradient during Critic optimization

ADD_OU_NOISE = True                 # Toggle Ornstein-Uhlenbeck noisy relaxation process
THETA = 0.15                        # k/gamma -> spring constant/friction coefficient [Ornstein-Uhlenbeck]
MU = 0.                             # x_0 -> spring length at rest [Ornstein-Uhlenbeck]
SIGMA = 0.2                         # root(2k_B*T/gamma) -> Stokes-Einstein for effective diffision [Ornstein-Uhlenbeck]
NOISE = 1.0                         # Initial Noise Amplitude
NOISE_REDUCTION = 0.995             # Noise amplitude decay ratio
```

### Results

With the afformentioned setup, the agent was able to successfully meet the functional specifications in 3193 episodes with an average score of [0.50 2.60] (see below):
```py
Episode 100     Average Score: 0.00 (nb of total steps=1466     noise=0.0006)
Episode 200     Average Score: 0.01 (nb of total steps=3066     noise=0.0000)
Episode 300     Average Score: 0.03 (nb of total steps=5204     noise=0.0000)
Episode 400     Average Score: 0.01 (nb of total steps=6840     noise=0.0000)
Episode 500     Average Score: 0.00 (nb of total steps=8291     noise=0.0000)
Episode 600     Average Score: 0.00 (nb of total steps=9711     noise=0.0000)
Episode 700     Average Score: 0.00 (nb of total steps=11131    noise=0.0000)
Episode 800     Average Score: 0.01 (nb of total steps=12726    noise=0.0000)
Episode 900     Average Score: 0.05 (nb of total steps=15057    noise=0.0000)
Episode 1000    Average Score: 0.02 (nb of total steps=17013    noise=0.0000)
Episode 1100    Average Score: 0.06 (nb of total steps=19768    noise=0.0000)
Episode 1200    Average Score: 0.05 (nb of total steps=22160    noise=0.0000)
Episode 1300    Average Score: 0.05 (nb of total steps=24842    noise=0.0000)
Episode 1400    Average Score: 0.05 (nb of total steps=27546    noise=0.0000)
Episode 1500    Average Score: 0.05 (nb of total steps=30057    noise=0.0000)
Episode 1600    Average Score: 0.05 (nb of total steps=32885    noise=0.0000)
Episode 1700    Average Score: 0.05 (nb of total steps=35902    noise=0.0000)
Episode 1800    Average Score: 0.06 (nb of total steps=39120    noise=0.0000)
Episode 1900    Average Score: 0.10 (nb of total steps=43527    noise=0.0000)
Episode 2000    Average Score: 0.06 (nb of total steps=46890    noise=0.0000)
Episode 2100    Average Score: 0.08 (nb of total steps=50792    noise=0.0000)
Episode 2200    Average Score: 0.07 (nb of total steps=54382    noise=0.0000)
Episode 2300    Average Score: 0.08 (nb of total steps=58002    noise=0.0000)
Episode 2400    Average Score: 0.11 (nb of total steps=62226    noise=0.0000)
Episode 2500    Average Score: 0.06 (nb of total steps=64770    noise=0.0000)
Episode 2600    Average Score: 0.05 (nb of total steps=67201    noise=0.0000)
Episode 2700    Average Score: 0.09 (nb of total steps=71145    noise=0.0000)
Episode 2800    Average Score: 0.12 (nb of total steps=75897    noise=0.0000)
Episode 2900    Average Score: 0.13 (nb of total steps=81294    noise=0.0000)
Episode 3000    Average Score: 0.22 (nb of total steps=89657    noise=0.0000)
Episode 3100    Average Score: 0.41 (nb of total steps=105475   noise=0.0000)
Environment solved in 3193 episodes with an Average Score of 0.50 2.60
```


![image](https://github.com/Ohara124c41/DRLND-Collaborate-n-Competition/blob/master/images/results.png?raw=true)


### Future Work

This section contains two additional algorithms that would vastly improve over the current implementation, namely TRPO and TD3. Such algorithms have been developed to improve over DQNs and DDPGs.

- [Distributed Distributional DDPG (D4PG)](https://arxiv.org/abs/1502.05477):
> We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement. By making several approximations to the theoretically-justified procedure, we develop a practical algorithm, called Trust Region Policy Optimization (TRPO). This algorithm is similar to natural policy gradient methods and is effective for optimizing large nonlinear policies such as neural networks. Our experiments demonstrate its robust performance on a wide variety of tasks: learning simulated robotic swimming, hopping, and walking gaits; and playing Atari games using images of the screen as input. Despite its approximations that deviate from the theory, TRPO tends to give monotonic improvement, with little tuning of hyperparameters.



- [Twin-Delay DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
> Twin Delayed Deep Deterministic policy gradient algorithm (TD3), an actor-critic algorithm which considers the interplay between function approximation error in both policy and value updates. We evaluate our algorithm on seven continuous control domains from OpenAI gym (Brockman et al., 2016), where we outperform the state of the art by a wide margin. TD3 greatly improves both the learning speed and performance of DDPG in a number of challenging tasks in the continuous control setting.  Our algorithm exceeds the performance of numerous state of the art algorithms. As our modifications are simple to implement, they can be easily added to any other actor-critic algorithm.


Further more, the [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment will be explored at a later time.

![image03](https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif)



## Additional References
_[1] Mordatch et al. (OpenAI), [Emergence of Grounded Compositional Language in Multi-Agent Populations](https://arxiv.org/abs/1703.04908)._

_[2] Lowe et al. (OpenAI), [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)._
