# Code Citations

## License: unknown
https://github.com/SarielMa/DQN_for_concurrency/tree/00ee7ab5d51f5d314e72773ecddbb6036fe2d9a9/DQN.py

```
policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.
```


## License: unknown
https://github.com/prestonyun/GymnasiumAgents/tree/71d686ff4706994246ab2fd16b9296efd74cb5e6/prioritized-dqn-agent.py

```
= torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states
```


## License: unknown
https://github.com/prestonyun/GymnasiumAgents/tree/71d686ff4706994246ab2fd16b9296efd74cb5e6/rldreamer/gymtest3.py

```
.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch
```


## License: unknown
https://github.com/Ahmedhossam36/adaptive-traffic-control-system-RL/tree/b597a15878f8af9a5fbe027a6f1f99215db89e7f/multidiscreteDQN.py

```
.to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1)
```


## License: unknown
https://github.com/ErikDeBruijn/cubeRL/tree/5b0ee79048c56cd65ffce2b042de497b85834ee3/dqnagent.py

```
):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(
```

