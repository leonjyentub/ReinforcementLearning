import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import gymnasium as gym

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 有那些遊戲環境可以使用，請參考
# https://www.gymlibrary.dev/environments/atari/complete_list/
# 這支程式是用來玩 Atari 遊戲 Pong 的強化學習範例
# https://www.gymlibrary.dev/environments/atari/pong/
'''
install: pip install ale-py

Once installed you can import the native ALE interface as ale_py
'''
import ale_py

gym.register_envs(ale_py)

def preprocess_frame(frame): 
    bkg_color = np.array([144, 72, 17])
    img = np.mean(frame[34:-16:2,::2]-bkg_color, axis=-1)/255.
    resized_image = img
    return resized_image

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    stack_size = 4
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((80,80), dtype=np.uint8) for i in range(stack_size)], maxlen=4)
        # Because we're in a new episode, copy the same frame 4x
        for i in range(stack_size):
            stacked_frames.append(frame) 
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2).transpose(2, 0, 1)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2).transpose(2, 0, 1) 
    return stacked_state, stacked_frames

    
class DQNetwork(nn.Module):
    def __init__(self, states, action_size):
        super(DQNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, (8, 8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, action_size)
        
    def forward(self, state): 
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Agent():
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(0)

        ## hyperparameters
        self.buffer_size = 1000
        self.batch_size = 32
        self.gamma = 0.99
        self.lr = 0.0001
        self.update_every = 4
        self.update_every_target = 1000 
        self.learn_every_target_counter = 0
        # Q-Network
        self.local = DQNetwork(state_size, action_size).to(device)
        self.target = DQNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = deque(maxlen=self.buffer_size) 
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # Initialize time step (for updating every few steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.append(self.experience(state[None], action, reward, next_state[None], done))
        
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.sample_experiences()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.local.eval()
            with torch.no_grad():
                action_values = self.local(state)
            self.local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        self.learn_every_target_counter+=1
        states, actions, rewards, next_states, dones = experiences
       # Get expected Q values from local model
        Q_expected = self.local(states).gather(1, actions)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current state
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.learn_every_target_counter%1000 ==0:
            self.target_update() 
            
    def target_update(self):
        print('\ntarget updating')
        self.target.load_state_dict(self.local.state_dict())

    def sample_experiences(self):
        experiences = random.sample(self.memory, k=self.batch_size)        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)        
        return (states, actions, rewards, next_states, dones)
    
if __name__ == "__main__":
    env = gym.make('ALE/Pong-v5', render_mode='human')
    print(env.spec.id)
    print(env.observation_space)
    print(env.action_space)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)
    n_episodes=5000
    max_t=5000
    eps_start=1.0
    eps_end=0.02
    eps_decay=0.995
    scores = [] # list containing scores from each episode
    scores_window = deque(maxlen=100) # last 100 scores
    eps = eps_start
    stack_size = 4
    stacked_frames = deque([np.zeros((80,80), dtype=int) for i in range(stack_size)], maxlen=stack_size) 

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state, _ = state
        state, frames = stack_frames(stacked_frames, state, True)
        score = 0
        for i in range(max_t):
            env.render()
            action = agent.act(state, eps)
            next_state, reward, done, *_ = env.step(action)
            next_state, frames = stack_frames(frames, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score) # save most recent score
        scores.append(score) # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tReward {} \tAverage Score: {:.2f} \tEpsilon: {}'.format(i_episode,score,np.mean(scores_window), eps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} \tEpsilon: {}'.format(i_episode, np.mean(scores_window), eps))