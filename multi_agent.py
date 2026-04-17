# multi_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(AgentQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MixingNetwork(nn.Module):
    # This definition matches the one expecting '_fc' suffixes
    def __init__(self, num_agents, global_state_dim, mixing_embed_dim=64):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.global_state_dim = global_state_dim
        self.mixing_embed_dim = mixing_embed_dim

        self.hyper_w1_fc = nn.Linear(global_state_dim, num_agents * mixing_embed_dim)
        self.hyper_b1_fc = nn.Linear(global_state_dim, mixing_embed_dim)
        
        self.hyper_w2_fc = nn.Linear(global_state_dim, mixing_embed_dim * 1) 
        self.hyper_b2_fc = nn.Sequential(
            nn.Linear(global_state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_q_values, global_state):
        batch_size = agent_q_values.size(0)
        w1 = torch.abs(self.hyper_w1_fc(global_state))
        b1 = self.hyper_b1_fc(global_state)
        w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim) 
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)
        agent_q_values_reshaped = agent_q_values.unsqueeze(1) 
        hidden = torch.relu(torch.bmm(agent_q_values_reshaped, w1) + b1)
        w2 = torch.abs(self.hyper_w2_fc(global_state))
        b2 = self.hyper_b2_fc(global_state)
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1) 
        b2 = b2.view(batch_size, 1, 1) 
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.squeeze(2).squeeze(1)


class QMIXAgent:
    def __init__(self, individual_state_dim, action_dim, num_agents, global_state_dim,
                 gamma=0.99, lr=0.0005, batch_size=32, memory_size=50000,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 target_update_freq=200, mixing_embed_dim=64, rng_instance=None):

        self.individual_state_dim = individual_state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.global_state_dim = global_state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.agent_networks = nn.ModuleList([AgentQNetwork(individual_state_dim, action_dim).to(device) for _ in range(num_agents)])
        self.target_agent_networks = nn.ModuleList([AgentQNetwork(individual_state_dim, action_dim).to(device) for _ in range(num_agents)])
        self.mixing_network = MixingNetwork(num_agents, global_state_dim, mixing_embed_dim).to(device)
        self.target_mixing_network = MixingNetwork(num_agents, global_state_dim, mixing_embed_dim).to(device)
        all_params = []
        for net in self.agent_networks: all_params.extend(list(net.parameters()))
        all_params.extend(list(self.mixing_network.parameters()))
        self.optimizer = optim.Adam(all_params, lr=lr)
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0
        self.rng = rng_instance if rng_instance else random.Random()
        self._update_target_networks()
        self.set_train_mode()


    def _update_target_networks(self):
        for i in range(self.num_agents):
            self.target_agent_networks[i].load_state_dict(self.agent_networks[i].state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def remember(self, individual_obs, joint_actions, reward, next_individual_obs, global_state, next_global_state, done):
        ind_obs_np = [np.array(obs, dtype=np.float32) for obs in individual_obs]
        actions_np = [int(act) for act in joint_actions]
        reward_val = float(reward)
        next_ind_obs_np = [np.array(obs, dtype=np.float32) for obs in next_individual_obs]
        global_state_np = np.array(global_state, dtype=np.float32)
        next_global_state_np = np.array(next_global_state, dtype=np.float32)
        done_bool = bool(done)
        self.memory.append((ind_obs_np, actions_np, reward_val, next_ind_obs_np,
                            global_state_np, next_global_state_np, done_bool))

    def act_individual(self, individual_observation, agent_id):
        is_training = self.agent_networks[agent_id].training 
        if is_training and self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.action_dim - 1)
        if not (0 <= agent_id < self.num_agents):
            return self.rng.randint(0, self.action_dim - 1)
        obs_tensor = torch.FloatTensor(np.array(individual_observation, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.agent_networks[agent_id](obs_tensor)
        return torch.argmax(q_values).item()

    def get_joint_action(self, individual_observations_list):
        joint_action = []
        for i in range(self.num_agents):
            obs_i = individual_observations_list[i] if i < len(individual_observations_list) else np.zeros(self.individual_state_dim, dtype=np.float32)
            action = self.act_individual(obs_i, i)
            joint_action.append(action)
        return joint_action

    def replay(self):
        if not self.mixing_network.training or len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        ind_obs_b, joint_actions_b, rewards_b, next_ind_obs_b, global_states_b, next_global_states_b, dones_b = zip(*batch)
        rewards_t = torch.FloatTensor(rewards_b).to(device) 
        dones_t = torch.FloatTensor(dones_b).to(device)     
        global_states_t = torch.FloatTensor(np.array(global_states_b)).to(device)
        next_global_states_t = torch.FloatTensor(np.array(next_global_states_b)).to(device)
        chosen_action_qvals_list = []
        with torch.no_grad():
            next_max_qvals_list = []
        for i in range(self.num_agents):
            obs_i_batch = torch.FloatTensor(np.array([obs_tuple[i] for obs_tuple in ind_obs_b])).to(device)
            next_obs_i_batch = torch.FloatTensor(np.array([obs_tuple[i] for obs_tuple in next_ind_obs_b])).to(device)
            actions_i_batch = torch.LongTensor([act_tuple[i] for act_tuple in joint_actions_b]).unsqueeze(1).to(device)
            q_i_all = self.agent_networks[i](obs_i_batch) 
            chosen_action_q_i = q_i_all.gather(1, actions_i_batch).squeeze(1) 
            chosen_action_qvals_list.append(chosen_action_q_i)
            with torch.no_grad():
                next_q_i_target_all = self.target_agent_networks[i](next_obs_i_batch) 
                next_max_q_i = next_q_i_target_all.max(1)[0] 
            next_max_qvals_list.append(next_max_q_i)
        chosen_action_qvals_stacked = torch.stack(chosen_action_qvals_list, dim=1) 
        next_max_qvals_stacked = torch.stack(next_max_qvals_list, dim=1) 
        q_tot_current = self.mixing_network(chosen_action_qvals_stacked, global_states_t) 
        with torch.no_grad():
            q_tot_target_next = self.target_mixing_network(next_max_qvals_stacked, next_global_states_t) 
        td_target = rewards_t + (1 - dones_t) * self.gamma * q_tot_target_next 
        loss = nn.MSELoss()(q_tot_current, td_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 1.0) 
        self.optimizer.step()
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self._update_target_networks()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'agent_networks_state_dict': [net.state_dict() for net in self.agent_networks],
            'target_agent_networks_state_dict': [net.state_dict() for net in self.target_agent_networks],
            'mixing_network_state_dict': self.mixing_network.state_dict(),
            'target_mixing_network_state_dict': self.target_mixing_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter
        }
        torch.save(save_dict, path)
        print(f"QMIX Agent model saved to {path} (using current key format)")

    def _adapt_mixing_network_state_dict(self, old_state_dict):
        """
        Adapts an old mixing network state_dict (without '_fc' suffix)
        to the new format (with '_fc' suffix).
        """
        new_state_dict = {}
        key_map = {
            "hyper_w1.weight": "hyper_w1_fc.weight", "hyper_w1.bias": "hyper_w1_fc.bias",
            "hyper_b1.weight": "hyper_b1_fc.weight", "hyper_b1.bias": "hyper_b1_fc.bias",
            "hyper_w2.weight": "hyper_w2_fc.weight", "hyper_w2.bias": "hyper_w2_fc.bias",
            "hyper_b2.0.weight": "hyper_b2_fc.0.weight", "hyper_b2.0.bias": "hyper_b2_fc.0.bias",
            "hyper_b2.2.weight": "hyper_b2_fc.2.weight", "hyper_b2.2.bias": "hyper_b2_fc.2.bias",
        }
        has_old_keys = any(k in old_state_dict for k in key_map.keys())
        
        if not has_old_keys: # If it already has new keys or is completely different, return as is
            print("Mixing network state_dict does not appear to use the old key format (without '_fc'). Loading as is.")
            return old_state_dict

        print("Adapting mixing network state_dict from old key format (without '_fc') to new format (with '_fc').")
        for old_key, value in old_state_dict.items():
            new_key = key_map.get(old_key, old_key) # Get new key if mapped, else keep old_key
            new_state_dict[new_key] = value
        
        # Verify if all expected new keys are present after adaptation
        # This is a sanity check. The load_state_dict will do the strict check.
        # expected_new_keys = set(key_map.values())
        # if not expected_new_keys.issubset(new_state_dict.keys()):
        #     print(f"Warning: After adapting mixing network keys, some expected new keys are still missing.")
            
        return new_state_dict

    def load_model(self, path):
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return
        try:
            checkpoint = torch.load(path, map_location=device)
            
            # Load agent networks with backward compatibility for key name
            if 'agent_networks_state_dict' in checkpoint: 
                agent_networks_states = checkpoint['agent_networks_state_dict']
                target_agent_networks_states = checkpoint['target_agent_networks_state_dict']
                for i in range(self.num_agents):
                    self.agent_networks[i].load_state_dict(agent_networks_states[i])
                    self.target_agent_networks[i].load_state_dict(target_agent_networks_states[i])
                print("Loaded QMIX agent networks using key: 'agent_networks_state_dict'")
            elif f'agent_0_network_state_dict' in checkpoint: 
                print("Attempting to load QMIX agent networks with legacy indexed keys (e.g., 'agent_0_network_state_dict')...")
                for i in range(self.num_agents):
                    self.agent_networks[i].load_state_dict(checkpoint[f'agent_{i}_network_state_dict'])
                    self.target_agent_networks[i].load_state_dict(checkpoint[f'target_agent_{i}_network_state_dict'])
                print("Successfully loaded QMIX agent networks using legacy indexed keys.")
            else:
                raise KeyError("Checkpoint does not contain recognizable keys for agent networks ('agent_networks_state_dict' or 'agent_i_network_state_dict').")
            
            for i in range(self.num_agents): 
                self.agent_networks[i].to(device)
                self.target_agent_networks[i].to(device)

            # Adapt and load mixing network state_dict
            if 'mixing_network_state_dict' in checkpoint:
                mixing_sd = checkpoint['mixing_network_state_dict']
                adapted_mixing_sd = self._adapt_mixing_network_state_dict(mixing_sd)
                self.mixing_network.load_state_dict(adapted_mixing_sd)
            else:
                print("Warning: 'mixing_network_state_dict' not found in checkpoint.")

            if 'target_mixing_network_state_dict' in checkpoint:
                target_mixing_sd = checkpoint['target_mixing_network_state_dict']
                adapted_target_mixing_sd = self._adapt_mixing_network_state_dict(target_mixing_sd)
                self.target_mixing_network.load_state_dict(adapted_target_mixing_sd)
            else:
                print("Warning: 'target_mixing_network_state_dict' not found in checkpoint.")

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for param_group in self.optimizer.param_groups:
                    for p in param_group['params']:
                        if p in self.optimizer.state:
                            state_dict_opt = self.optimizer.state[p]
                            for k_opt, v_opt in state_dict_opt.items():
                                if isinstance(v_opt, torch.Tensor):
                                    if k_opt == 'step':
                                        state_dict_opt[k_opt] = v_opt.to('cpu')
                                    else:
                                        state_dict_opt[k_opt] = v_opt.to(device)
            else:
                print("Warning: Optimizer state not found in checkpoint. Optimizer not loaded.")

            self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
            self.learn_step_counter = checkpoint.get('learn_step_counter', 0)
            
            self.mixing_network.to(device)
            self.target_mixing_network.to(device)
            
            print(f"QMIX Agent model loaded from {path}")
            self.set_train_mode()

        except KeyError as e:
            print(f"Error loading QMIX model due to KeyError: {e}. Model file might be incompatible or corrupted.")
            if 'checkpoint' in locals() and isinstance(checkpoint, dict):
                print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
            raise 
        except RuntimeError as e: # Catch RuntimeError from load_state_dict
            print(f"Error loading QMIX model state_dict (likely for MixingNetwork): {e}")
            if 'checkpoint' in locals() and isinstance(checkpoint, dict):
                 if 'mixing_network_state_dict' in checkpoint:
                    print(f"Keys in mixing_network_state_dict from checkpoint: {list(checkpoint['mixing_network_state_dict'].keys())}")
                 if 'target_mixing_network_state_dict' in checkpoint:
                    print(f"Keys in target_mixing_network_state_dict from checkpoint: {list(checkpoint['target_mixing_network_state_dict'].keys())}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading QMIX model: {e}")
            raise 

    def set_eval_mode(self):
        self.epsilon = 0.0
        for agent_net in self.agent_networks: agent_net.eval()
        for target_agent_net in self.target_agent_networks: target_agent_net.eval()
        self.mixing_network.eval()
        self.target_mixing_network.eval()
        print("QMIX Agent set to evaluation mode (epsilon=0, all networks .eval()).")
        
    def set_train_mode(self):
        for agent_net in self.agent_networks: agent_net.train()
        for target_agent_net in self.target_agent_networks: target_agent_net.train()
        self.mixing_network.train()
        self.target_mixing_network.train()
        print("QMIX Agent set to training mode (all networks .train()).")