import gym 
gym.logger.set_level(40)
import numpy as np
import matplotlib.pyplot as plt

def main(): 
  ''' 
  Runs policy gradient on Acrobot-v1 and CartPole-v1 environments.
  '''

  bot = gym.make('Acrobot-v1')
  policy_gradient(bot, 100, max_traj=200)
  bot.close()

  pole = gym.make('CartPole-v1')
  policy_gradient(pole, 500, traj_sample=20)
  pole.close()


def policy_gradient(env, epoch=100, alpha=0.001, traj_sample=10,\
                    max_traj=500, display=True):
  ''' 
  Uses policy gradient with stochastic-esque ascent on environment. 
  Discrete action space and finite features of observation space 
  required.

  Parameters
  ----------
  env : 
      Environment to perform algorithm on. 
  epoch : int, optional, default: 100
      Number of weight adjustments.
  alpha : float, optional, default: 0.001
      Training rate of gradient ascent.
  traj_sample : int, optional, default: 10
      Number of trajectory samples per epoch. 
  max_traj : int, optional, default: 500
      Maximum trajectory length recorded.
  display : bool, optional, default: True
      Display training data and performance over three trials. 
  
  Returns
  -------
  policy_gradient : ndarray of floats
      Returns ndarray containing  weights to softmax based policy.
  ''' 

  # Dimension variables 
  state_dim = np.prod(env.observation_space.shape)
  action_dim = env.action_space.n

  # Hold softmax weights, xavier init (mean=0, var=(1/in_units))
  weights = np.random.normal(0, \
    1 / np.sqrt(state_dim), \
    (action_dim, state_dim))

  # Graphing purposes
  if display:
    epochs = [i for i in range(epoch)]
    rewards = [0] * epoch 

  # Run epochs
  for ep in range(epoch):
    traj_reward = 0 
    grad = np.zeros_like(weights)
    for tau in range(traj_sample):
      state = env.reset()
      # Step of trajectory 
      for t in range(max_traj):
        # Apply softmax policy to state
        prob_actions = _softmax(np.matmul(weights, state.reshape(state_dim, 1)))
        # Choose action based on probablities 
        action_choice = np.random.choice(action_dim, p=prob_actions.flatten())
        # Apply action to environment
        state, reward, done, _ = env.step(action_choice)

        if done: 
          break

        traj_reward += reward

        # Apply one-hot 
        prob_actions = -prob_actions
        prob_actions[action_choice] += 1
        # Calculate gradient 
        grad += np.matmul( \
          (prob_actions * reward).reshape(action_dim, 1), \
          state.reshape(1, state_dim))

    # Apply gradient (see derivations for subtraction) 
    weights -= alpha * grad / traj_sample

    rewards[ep] = traj_reward / traj_sample

  # Plot performance over epochs
  if display:
    _disp(env, alpha, traj_sample, max_traj, epochs, rewards, weights)


def _disp(env, alpha, traj_sample, max_traj, epochs, rewards, weights):
  ''' 
  Displays reward obtained during training and 3 test runs.
  '''

  # Plot the rewards over epochs
  plt.plot(epochs, rewards, 'b')
  plt.xlabel('Number of Epochs')
  plt.ylabel('Mean Reward per Trajectory')
  plt.title(f'Policy Gradient Training Performance over Epochs ({alpha} α,'\
    f' {traj_sample} τ)')
  plt.show()

  # Run three trials 
  for e in range(1, 4): 
    state = env.reset()
    # Step of trajectory 
    for t in range(max_traj):
      env.render()
      # Apply softmax policy to state
      prob_actions = _softmax(np.matmul(weights, state))
      # Choose action based on probablities 
      action_choice = np.random.choice(env.action_space.n, p=prob_actions)
      # Apply action to environment
      state, reward, done, _ = env.step(action_choice)
      if done:
        print(f'Trial {e} lasted {t} steps.')
        break


def _softmax(x): 
  '''
  Softmax function over array. 
  '''

  return np.exp(x - np.amax(x)) / np.sum(np.exp(x - np.amax(x)))

if __name__ == "__main__":
  main()

