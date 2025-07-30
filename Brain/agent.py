import numpy as np
from .model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator
import torch
from .replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax
import torch.nn.functional as F

class SACAgent:
    def __init__(self,
                 p_z,
                 **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(self.config["seed"])
        self.policy_network = PolicyNetwork(n_states=self.n_states + self.n_skills,
                                            n_actions=self.config["n_actions"],
                                            n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                          n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_target_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                                 n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        self.hard_update_target_network()

        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

    def choose_action(self, state, greedy=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            idx, _ = self.policy_network.sample(state, greedy)
        return int(idx.cpu().item())          # integer that Acrobot understands

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        z     = torch.LongTensor([z]).to("cpu")
        done  = torch.BoolTensor([done]).to("cpu")
        action= torch.LongTensor([action]).to("cpu")     # ♦ change ♦
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1,1).long().to(self.device)   # index
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)

        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:          # unchanged
            return None
        batch       = self.memory.sample(self.batch_size)
        states, zs, dones, actions, next_states = self.unpack(batch)

        # 1. π, log π, Q(s,·)
        probs, log_probs = self.policy_network(states)
        q1_all           = self.q_value_network1(states)
        q2_all           = self.q_value_network2(states)
        q_min_det        = torch.min(q1_all, q2_all).detach()   # already detached

        # DETACH policy terms for the value target
        probs_det       = probs.detach()
        log_probs_det   = log_probs.detach()
        # 2. V target for current state   V(s) = Σ_a π(a|s)(Q_min - α log π)
        target_v = (probs_det * (q_min_det - self.config["alpha"] * log_probs_det)).sum(dim=-1, keepdim=True)
        value     = self.value_network(states)
        value_loss= self.mse_loss(value, target_v)

        # 3. intrinsic reward from discriminator (unchanged)
        logits      = self.discriminator(states[:, :self.n_states])
        p_z         = torch.from_numpy(self.p_z).to(self.device).gather(-1, zs)
        logq_z_ns   = F.log_softmax(logits, dim=-1)
        rewards     = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z+1e-6)

        # 4. Q-targets               Q_targ = r_int * β + γ (1-d) V'(s')
        with torch.no_grad():
            next_v = self.value_target_network(next_states)
            target_q = self.config["reward_scale"]*rewards.float() + \
                    self.config["gamma"] * next_v * (~dones)

        # 5. Chosen-action Q-values
        q1 = q1_all.gather(1, actions)
        q2 = q2_all.gather(1, actions)
        q1_loss = self.mse_loss(q1, target_q)
        q2_loss = self.mse_loss(q2, target_q)

        # 6. Policy loss  J_π = E_s [ Σ_a π(a|s)(α log π - Q_min) ]
        policy_loss = (probs * (self.config["alpha"] * log_probs - q_min_det)).sum(dim=-1).mean()
        states_only, _ = torch.split(states, [self.n_states, self.n_skills], dim=-1)
        logits = self.discriminator(states_only)
        discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

        self.policy_opt.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value_opt.step()

        self.q_value1_opt.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q_value1_opt.step()

        self.q_value2_opt.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.q_value2_opt.step()

        self.discriminator_opt.zero_grad()
        discriminator_loss.backward()
        self.discriminator_opt.step()

        self.soft_update_target_network(self.value_network, self.value_target_network)

        return -discriminator_loss.item()

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)
