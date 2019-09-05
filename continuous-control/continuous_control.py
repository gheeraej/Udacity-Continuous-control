import torch
import numpy as np
import pandas as pd
from collections import deque
from unityagents import UnityEnvironment
import random
import matplotlib.pyplot as plt
%matplotlib inline

from ddpg_agent import Agent


def plot_scores(scores, rolling_window=10, save_fig=False):
    """Plot scores and optional rolling mean using specified window."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'scores')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);

    if save_fig:
        plt.savefig(f'figures_scores.png', bbox_inches='tight', pad_inches=0)
        

def train_ddpg(env, agent, num_agents, save_or_load_path, n_episodes=10000, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            
            env_info = env.step(actions)[brain_name]   
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if any(dones):
                break 
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), save_or_load_path)
        torch.save(agent.critic_local.state_dict(), save_or_load_path)
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            plot_scores(scores)
        if np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - print_every, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), save_or_load_path)
            torch.save(agent.critic_local.state_dict(), save_or_load_path)
            break
            
    return scores


### Test the agent
def test_ddpg(env, agent, num_agents, max_t=1000):
    brain_name = env.brain_names[0]
    
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations  
    
    score = np.zeros(num_agents)
    for t in range(max_t):
        action = agent.act(state, add_noise=False)
        
        env_info = env.step(actions)[brain_name]   
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished

        agent.step(states, actions, rewards, next_states, dones)
        states = next_states
        score += rewards
        if any(dones):
            break
    print("Score of this episode is: %.2f" % np.mean(score))         
                
                
### Launcher function
def launch(app_path, train_or_test, save_or_load_path, hyper_file):
    env = UnityEnvironment(file_name=app_path)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
        
    # reset the environment
    env_info = env.reset(train_mode=train_or_test)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    agent = Agent(state_size=state_size, action_size=action_size, n_agents=num_agents, random_seed=42)    
    
    if train_or_test:
        if hyper_file is None:
            scores = train_ddpg(env, agent, num_agents, save_or_load_path)
        else:
            with open(hyper_file) as f:
                variables = json.load(f)
                if len(list(set(variables.keys()) & set(["n_episodes", "max_t", "print_every"]))) != 3:
                    print("Parameters file is not well specified")
                    pass
                else:
                    scores = train_ddpg(env, agent, num_agents, save_or_load_path, variables["n_episodes"], variables["max_t"], variables["print_every"])

        plot_scores(scores, True)
        
    else:
        agent.qnetwork_local.load_state_dict(torch.load(save_or_load_path))
        test_ddpg(env, agent, num_agent)
    
    env.close()


if __name__=="__main__":
    if len(sys.argv)<4:
        print("Argument 1 is mandatory and must be the path of the Reacher env")
        print("Argument 2 is mandatory and is 1 for train and 0 for test")
        print("Argument 3 is mandatory and is the path of the file to save weights (in train) or to load weights (in test)")
        pass
    if len(sys.argv)<5:
        print("Argument 4 represents the hyper-parameter file (defaut to 'None')")
        launch(sys.argv[1], bool(int(sys.argv[2])), sys.argv[3], None)
    else:
        launch(sys.argv[1], bool(int(sys.argv[2])), sys.argv[3], sys.argv[4])
    