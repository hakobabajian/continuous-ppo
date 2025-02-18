# Continuous Proximal Policy Optimization 
## Objective
This project aims to train a continuous ppo, a reinforement learning model to control the pitch authority of an aircraft in the popular aerospace simulator, Kerbal Space Program. The goal is to train the model such that it can pilot the aircraft to take-off from the runway, and maintain an a reasonable altitude without going out of bounds for 30 seconds.

#### Dependencies
This project uses the following python libraries:
- pytorch: for developing and training neural networks
- krpc: for programmatic control of aircraft in the aerospace simulation software

## Technical Overview
The following is a very high-level explanation of the reinforcement learning / machine learning methods and practices used in this project.
### Actor-Critic Networks
Actor-Critic networks are a pair of two distinct neural networks which operate in tandem to optimize the training protocol. Both networks receive the same inputs, their distinction is defined by what they output:
- The Actor network outputs the chosen action of the machine learning model
- The Critic network outputs an evaluation of the previous action taken by the Actor network

There are many reinforcement learning models which make use of Actor-Critic networks, one of which is Proximal Policy Optimization (PPO)
### Proximal Policy Optimization (PPO)
PPO's are a reinforcement learning algorithm which aim to improve upon a common weakness of Actor-Critic networks: being highly sensitive to perturbations or massive overcorrections in learned changes to policy. This can cause premature convergence onto a policy, where the networks gets stuck and continually chooses the same action (often a very incorrect action).

PPO's attempt to avoid perturabtions by limiting the number of updates to the networks. Updates are done less frequently, and more comprehensively by training with several averaged data samples in a process called mini-batch stochastic gradient ascent.

Another technique used in the implementation of PPO in this project was refining the hyper-parameter known as the entropy coefficient.

### Entropy Regularization
As previously stated, a common problem in deep reinforcement learning is premature policy convergence, and this project was no exception. Sufficiently raising the agent's entropy coefficient was essential in having the model continue to explore different trajectories and learn from them, over the thousands of learning iterations.

## Continuous PPO
Both the Actor and Critic neural networks in this project are comprised of an input layer, two hidden layers, and an output layer. This PPO is defined as "continuous" (as opposed to "discrete"), because its action (output) is a float value (0,1). 

The Actor nework's input and output layers were defined respectively:
- Observation Space: [speed, altitude, vertical velocity]
- Action Space: [float] (0,1)
#### Methodology
In between simulation environment steps, the agent gets a new iteration of observations, chooses an action, and intermittently takes a sample of the previous few iterations' data and learns. This repeats until the end of the training iteration.

### Training Environment
The training environment is designed via programmatic access to an aerospace simulator, where training iterations each last a maximum 30 seconds. There are several out-of-bounds conditions which immediately terminate the training iteration. The training environment begins with the aircraft on the runway with no speed. A conventional control algorithm gets the aircraft to throttle up and then maintain a target speed. The tear-down/restart of training environment happens programmaticly between learning iterations.
#### Environment Start
The training environment uses numeric methods to handle control authority updates and new observations. Initial observations are returned to the agent.
#### Environment Step
Each simulation step is defined at a minimum based on the environment class constructor: `self.t = time_step`. Each environment step does the following in order:
- makes pitch adjustment based on action chosen by agent
- defines the numeric differential value `self.t` based on how much time has passed since the last step and, if necessary, allows time to pass
- makes throttle adjustment with control methods to maintain constant aircraft speed
- calculates a new observation array
- appends reward points to reward score
- checks for learning iteration termination conditions
- returns observation array, reward score, and done boolean

#### Reward function

## Results



