# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .memory_policy import MemoryPolicy
from .sa_schedules import ThreeTimeScale


class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class GNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DeepTPI:
    """
    Deep variant: q_u(theta,a; w), g_u(h,a; φ), σ_u(h) mirror ascent.
    Local, constant-time per (u).
    """

    def __init__(
        self,
        dag,
        obs_dim_per_u: Dict[int, int],
        hist_dim_per_u: Dict[int, int],
        n_actions_per_u: Dict[int, int],
        timescale: ThreeTimeScale,
        m_memory: int,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.dag = dag
        self.obs_dim = obs_dim_per_u
        self.hist_dim = hist_dim_per_u
        self.n_actions = n_actions_per_u
        self.ts = timescale
        self.m = m_memory
        self.device = device
        torch.manual_seed(seed)
        # per-substage models
        self.q_nets: Dict[int, QNet] = {}
        self.g_nets: Dict[int, GNet] = {}
        self.q_opts: Dict[int, optim.Optimizer] = {}
        self.g_opts: Dict[int, optim.Optimizer] = {}
        self.policies: Dict[int, MemoryPolicy] = {}

        for u in self.dag.reverse_order():
            uidx = u.idx
            qa = QNet(self.obs_dim[uidx], self.n_actions[uidx]).to(self.device)
            ga = GNet(self.hist_dim[uidx], self.n_actions[uidx]).to(self.device)
            self.q_nets[uidx] = qa
            self.g_nets[uidx] = ga
            self.q_opts[uidx] = optim.Adam(qa.parameters(), lr=self.ts.alpha)
            self.g_opts[uidx] = optim.Adam(ga.parameters(), lr=self.ts.beta)
            self.policies[uidx] = MemoryPolicy(self.n_actions[uidx], gamma_init=self.ts.gamma)

    def step(self, sample_dict: Dict[int, Dict[str, Any]], k: int):
        ts = self.ts.decay(k)
        for u in self.dag.reverse_order():
            uidx = u.idx
            if uidx not in sample_dict:
                continue
            s = sample_dict[uidx]
            theta = s["theta_vec"]  # np.array
            a = s["a"]
            r = s["r"]
            theta_next = s["theta_next_vec"]
            h_vec = s["h_vec"]
            h_next_vec = s["h_next_vec"]
            a_next = s["a_next"]
            succ_idx = u.successor

            # critic
            q_net = self.q_nets[uidx]
            g_net = self.g_nets[uidx]
            q_opt = self.q_opts[uidx]
            g_opt = self.g_opts[uidx]

            q_net.train()
            g_net.train()

            theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = q_net(theta_t)
            q_a = q_vals[0, a]

            with torch.no_grad():
                target = torch.tensor([r], dtype=torch.float32, device=self.device)
                if succ_idx is not None and succ_idx in self.q_nets:
                    q_succ = self.q_nets[succ_idx]
                    theta_next_t = torch.tensor(theta_next, dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_next_vals = q_succ(theta_next_t)[0]
                    target = target + q_next_vals[a_next]

            loss_q = 0.5 * (q_a - target) ** 2
            q_opt.zero_grad()
            loss_q.backward()
            # gradient clipping for boundedness
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
            q_opt.step()

            # aggregator
            h_t = torch.tensor(h_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            g_vals = g_net(h_t)[0]
            # we want g(h,a) -> q(theta,a)
            q_a_detach = q_a.detach()
            loss_g = 0.5 * (g_vals[a] - q_a_detach) ** 2
            g_opt.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(g_net.parameters(), max_norm=5.0)
            g_opt.step()

            # mirror ascent
            g_np = g_vals.detach().cpu().numpy()
            self.policies[uidx].mirror_update(uidx, tuple(h_vec.tolist()), g_np, gamma=ts.gamma)

    def act(self, uidx: int, h_vec, greedy: bool = False) -> int:
        if greedy:
            return self.policies[uidx].greedy(uidx, tuple(h_vec.tolist()))
        return self.policies[uidx].sample(uidx, tuple(h_vec.tolist()), np.random.default_rng())