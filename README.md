# Solutions to OpenAI Gym problems
Implementations from scratch using only NumPy for vector computations. Running files as provided will display training information over epochs and several rendered trials of the solution in practice.

#### policy-grad-derivations.pdf
Derivations of softmax policy gradient with expected value objective function. Includes review of random basic RL concepts. Assistance with general form of policy gradient derivation [from here](http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_4_policy_gradient.pdf). 
#### policy_grad.py
Implementation of softmax policy gradient on CartPole-v1 and Acrobot-v1. Training performance and several trials on trained model displayed when running file as is. Default arguments run CartPole-v1 for 500 epochs, with 20 random trajectories per epoch. Acrobot-v1 runs for 100 epochs, with 10 random trajectories per epoch. 
#### rand_search.py
Implementation of random search on above environments. Randomly initialized set of weights, vector for each action in the dimension of the state space. Training performance and several trials on trained model displayed when running file as is. Set of weights obtaining the maximum reward across 10 trajectories kept as the best performing model. 
