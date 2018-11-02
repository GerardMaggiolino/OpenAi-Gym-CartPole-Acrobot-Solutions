import gym 
gym.logger.set_level(40)
import numpy as np
import matplotlib.pyplot as plt

def main(): 
  ''' 
  Run random search on CartPole-v1 and Acrobot-v1 environment. 
  '''

  pole = gym.make('CartPole-v1')
  linear_random_search(pole, 200, 500, min_reward=499)
  pole.close()

  
  bot = gym.make('Acrobot-v1')
  linear_random_search(bot, 50, 200)
  bot.close()

def linear_random_search(env, epoch,  max_traj=500, display=True, \
      min_reward=None): 
  ''' 
  Performs random search with linear decision boundries on environment.

  Parameters
  ----------
  env : 
      Environment to perform algorithm on. 
  epoch : int
      Number of randomized weights to be tested
  display : bool, optional, default: True
      Display training data and performance over three trials. 
  min_reward : int, default: None
      Minimum reward obtained over 20 trials to consider the problem
      solved. Default is to complete all epochs.

  Returns
  -------
  linear_random_search : None
  ''' 

  # Store weight information
  state_size = np.product(env.observation_space.shape)
  action_size = env.action_space.n
  best_weights = np.random.rand(action_size, state_size) * 2 - 1
  epoch_weights = best_weights.copy()

  # Graphing and tracking
  best_epoch = 0
  rewardList = list()
  epochs = list()

  epochs.append(0)
  rewardList.append(_random_search_trial(env, epoch_weights, max_traj))
  epoch_weights = np.random.rand(action_size, state_size) * 2 - 1
  for ep in range(1, epoch):
    epochs.append(ep)
    epoch_reward = _random_search_trial(env, epoch_weights, max_traj)

    # Save best weights
    if rewardList[best_epoch] <= epoch_reward:
      best_weights = epoch_weights
      best_epoch = ep
      rewardList.append(epoch_reward)
    else: 
      rewardList.append(rewardList[best_epoch])
    
    # Solved by finding proper weights
    if min_reward is not None and rewardList[best_epoch] > min_reward:
      break

    epoch_weights = np.random.rand(action_size, state_size) * 2 - 1

  # Plot performance over epochs
  if display: 
    plt.plot(epochs, rewardList, 'g')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Best Minimum Reward over 10 Trials')
    plt.title('Training Performance of Random Search')
    plt.show()

    # Show with best weights for 3 trials 
    for t in range(1, 4):
      observation = env.reset()
      for s in range(500):
        env.render()
        action = np.argmax(np.matmul(best_weights, \
          observation.reshape(np.prod(env.observation_space.shape), 1)))
        observation, reward, done, _ = env.step(action) 
        if done:
          print(f'Trial {t} ended at {s} steps with Random Search')
          break


def _random_search_trial(env, epoch_weights, max_traj): 
  ''' 
  Returns lowest reward in of a weight set over 10 trials.

  Parameters 
  ----------
  env :
      Environment to perform algorithm on. 
  epoch_weights : numpy.ndarray
      Contains the weights used as a linear decision boundary.

  Returns
  ------- 
  _random_search_trial : int:
      The lowest reward obtained by epoch_weights over 10 trials.  
  '''

  trials = 10
  rewards = np.zeros(trials)
  for t in range(trials): 
    observation = env.reset()
    for _ in range(max_traj):
      action = np.argmax(np.matmul(epoch_weights, \
        observation.reshape(np.prod(env.observation_space.shape), 1)))
      observation, reward, done, _ = env.step(action) 
      rewards[t] += reward
      if done:
        break 
  return np.amin(rewards)


if __name__ == "__main__":
  main()

