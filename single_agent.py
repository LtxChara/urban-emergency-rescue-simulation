# single_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random 
from collections import deque
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN_Network, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001,
                 batch_size=64, memory_size=50000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999,
                 target_update_freq=200, rng_instance=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=memory_size)

        self.model = DQN_Network(state_dim, action_dim).to(device)
        self.target_model = DQN_Network(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        self.rng = rng_instance if rng_instance else random.Random()

        self.update_target_model()
        # Initialize models in training mode by default
        self.model.train()
        self.target_model.train()


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        state_np = np.array(state, dtype=np.float32)
        next_state_np = np.array(next_state, dtype=np.float32)
        action_val = int(action)
        reward_val = float(reward)
        done_bool = bool(done)
        self.memory.append((state_np, action_val, reward_val, next_state_np, done_bool))

    def act(self, state):
        if self.model.training and self.rng.random() < self.epsilon: # Only explore if in training mode
            return self.rng.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad(): # Always no_grad for inference part of act
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if not self.model.training or len(self.memory) < self.batch_size : # Only replay if in training mode
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

        current_q_values = self.model(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_max_q_values = self.target_model(next_states_t).max(1)[0].unsqueeze(1)

        target_q_values = rewards_t + (1 - dones_t) * self.gamma * next_max_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon)

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_model()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter
        }, path)
        print(f"DQN Agent model saved to {path}")

    def load_model(self, path):
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return
        try:
            checkpoint = torch.load(path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
            self.learn_step_counter = checkpoint.get('learn_step_counter', 0)
            
            self.model.to(device)
            self.target_model.to(device)

            # Optimizer state loading (as previously corrected)
            for param_group in self.optimizer.param_groups:
                for p in param_group['params']:
                    if p in self.optimizer.state:
                        state = self.optimizer.state[p]
                        for k, v_opt in state.items():
                            if isinstance(v_opt, torch.Tensor):
                                if k == 'step':
                                    state[k] = v_opt.to('cpu')
                                else:
                                    state[k] = v_opt.to(device)
            
            print(f"DQN Agent model loaded from {path}")
            # Set to train mode by default after loading, set_eval_mode will override if called
            self.model.train()
            self.target_model.train()

        except Exception as e:
            print(f"Error loading DQN model: {e}")
            raise # Re-raise exception to see details

    def set_eval_mode(self):
        self.epsilon = 0.0
        if hasattr(self, 'model'):
            self.model.eval()
        if hasattr(self, 'target_model'): # Target model also used for Q-value calculation in some setups
            self.target_model.eval()
        print("DQN Agent set to evaluation mode (epsilon=0, model.eval()).")

    def set_train_mode(self):
        # self.epsilon = self.epsilon_start # Or restore last training epsilon
        if hasattr(self, 'model'):
            self.model.train()
        if hasattr(self, 'target_model'):
            self.target_model.train()
        print("DQN Agent set to training mode (model.train()).")