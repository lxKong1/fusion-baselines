"""Network components adapted from FUSION CPQ; see LICENSE."""
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F


@dataclass
class CPQMetaDriveConfig:
    a_hidden_sizes: list = field(default_factory=lambda: [256, 256])
    c_hidden_sizes: list = field(default_factory=lambda: [256, 256])
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = .99
    tau: float = .005
    beta: float = .5
    num_q: int = 2
    num_qc: int = 2
    qc_scalar: float = 1.5
    episode_len: int = 1000
    cost_limit: int = 10
    batch_size: int = 512
    update_steps: int = 100_000
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-5
    vae_lr: float = 1e-4


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Creates a multi-layer perceptron with the specified sizes and activations.

    Args:
        sizes (list): A list of integers specifying the size of each layer in the MLP.
        activation (nn.Module): The activation function to use for all layers except the output layer.
        output_activation (nn.Module): The activation function to use for the output layer. Defaults to nn.Identity.

    Returns:
        nn.Sequential: A PyTorch Sequential model representing the MLP.
    """

    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        layers += [layer, act()]
    return nn.Sequential(*layers)

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):
    '''
    A MLP Gaussian actor, can also be used as a deterministic actor

    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
    '''

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self,
                obs,
                deterministic=False,
                with_logprob=True,
                with_distribution=False,
                return_pretanh_value=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # for BEARL only
        if return_pretanh_value:
            return torch.tanh(pi_action), pi_action

        pi_action = torch.tanh(pi_action)

        if with_distribution:
            return pi_action, logp_pi, pi_distribution
        return pi_action, logp_pi


class EnsembleQCritic(nn.Module):
    '''
    An ensemble of Q network to address the overestimation issue.

    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        num_q (float): The number of Q networks to include in the ensemble.
    '''

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"
        self.q_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
            for i in range(num_q)
        ])

    def forward(self, obs, act=None):
        # Squeeze is critical to ensure value has the right shape.
        # Without squeeze, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        data = obs if act is None else torch.cat([obs, act], dim=-1)
        return [torch.squeeze(q(data), -1) for q in self.q_nets]

    def predict(self, obs, act):
        q_list = self.forward(obs, act)
        qs = torch.vstack(q_list)  # [num_q, batch_size]
        return torch.min(qs, dim=0).values, q_list

    def loss(self, target, q_list=None):
        losses = [((q - target)**2).mean() for q in q_list]
        return sum(losses)


class VAE(nn.Module):
    """
    Variational Auto-Encoder

    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_size (int): The number of hidden units in the encoder and decoder networks.
        latent_dim (int): The dimensionality of the latent space.
        act_lim (float): The upper limit of the action space.
        device (str): The device to use for computation (cpu or cuda).
    """
    def __init__(self, obs_dim, act_dim, hidden_size, latent_dim, act_lim, device="cpu"):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d1 = nn.Linear(obs_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, act_dim)

        self.act_lim = act_lim
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, obs, act):
        z = F.relu(self.e1(torch.cat([obs, act], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(obs, z)
        return u, mean, std

    def decode(self, obs, z=None):
        if z is None:
            z = torch.randn((obs.shape[0], self.latent_dim)).clamp(-0.5, 0.5).to(self.device)

        a = F.relu(self.d1(torch.cat([obs, z], 1)))
        a = F.relu(self.d2(a))
        return self.act_lim * torch.tanh(self.d3(a))


class CPQ(nn.Module):
    """
    Constraints Penalized Q-Learning (CPQ)

    Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list): List of integers specifying the sizes
                               of the layers in the actor network.
        c_hidden_sizes (list): List of integers specifying the sizes
                               of the layers in the critic network.
        vae_hidden_sizes (int): Number of hidden units in the VAE.
        sample_action_num (int): Number of action samples to draw.
        gamma (float): Discount factor for the reward.
        tau (float): Soft update coefficient for the target networks.
        beta (float): Weight of the KL divergence term.
        num_q (int): Number of Q networks in the ensemble.
        num_qc (int): Number of cost Q networks in the ensemble.
        qc_scalar (float): Scaling factor for the cost critic threshold.
        cost_limit (int): Upper limit on the cost per episode.
        episode_len (int): Maximum length of an episode.
        device (str): Device to run the model on (e.g. 'cpu' or 'cuda:0').
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 a_hidden_sizes: list = [128, 128],
                 c_hidden_sizes: list = [128, 128],
                 vae_hidden_sizes: int = 64,
                 sample_action_num: int = 10,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 beta: float = 1.5,
                 num_q: int = 1,
                 num_qc: int = 1,
                 qc_scalar: float = 1.5,
                 cost_limit: int = 10,
                 episode_len: int = 300,
                 device: str = "cpu"):

        super().__init__()
        self.a_hidden_sizes = a_hidden_sizes
        self.c_hidden_sizes = c_hidden_sizes
        self.vae_hidden_sizes = vae_hidden_sizes
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.cost_limit = cost_limit
        self.num_q = num_q
        self.num_qc = num_qc
        self.qc_scalar = qc_scalar
        self.sample_action_num = sample_action_num

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = self.action_dim * 2
        self.episode_len = episode_len
        self.max_action = max_action

        self.device = device

        ################ create actor critic model ###############
        self.actor = SquashedGaussianMLPActor(self.state_dim, self.action_dim,
                                              self.a_hidden_sizes,
                                              nn.ReLU).to(self.device)
        self.critic = EnsembleQCritic(self.state_dim,
                                      self.action_dim,
                                      self.c_hidden_sizes,
                                      nn.ReLU,
                                      num_q=self.num_q).to(self.device)
        self.vae = VAE(self.state_dim, self.action_dim, self.vae_hidden_sizes,
                       self.latent_dim, self.max_action, self.device).to(self.device)
        self.cost_critic = EnsembleQCritic(self.state_dim,
                                           self.action_dim,
                                           self.c_hidden_sizes,
                                           nn.ReLU,
                                           num_q=self.num_qc).to(self.device)
        self.log_alpha = torch.tensor(0.0, device=self.device)

        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()
        self.critic_old = deepcopy(self.critic)
        self.critic_old.eval()
        self.cost_critic_old = deepcopy(self.cost_critic)
        self.cost_critic_old.eval()

        # set critic and cost critic threshold
        self.q_thres = cost_limit * (1 - self.gamma**self.episode_len) / (
            1 - self.gamma) / self.episode_len
        self.qc_thres = qc_scalar * self.q_thres

    def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """
        Softly update the parameters of target module
        towards the parameters of source module.
        """
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def _actor_forward(self,
                       obs: torch.tensor,
                       deterministic: bool = False,
                       with_logprob: bool = True):
        """
        Return action distribution and action log prob [optional].
        """
        a, logp = self.actor(obs, deterministic, with_logprob)
        return a * self.max_action, logp

    def vae_loss(self, observations, actions):
        recon, mean, std = self.vae(observations, actions)
        recon_loss = nn.functional.mse_loss(recon, actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        loss_vae = recon_loss + self.beta * KL_loss

        self.vae_optim.zero_grad()
        loss_vae.backward()
        self.vae_optim.step()
        stats_vae = {"loss/loss_vae": loss_vae.item()}
        return loss_vae, stats_vae

    def critic_loss(self, observations, next_observations, actions, rewards, done):
        _, q_list = self.critic.predict(observations, actions)
        # Bellman backup for Q functions
        with torch.no_grad():
            next_actions, _ = self._actor_forward(next_observations, False, True)
            q_targ, _ = self.critic_old.predict(next_observations, next_actions)
            qc_targ, _ = self.cost_critic_old.predict(next_observations, next_actions)
            # Constraints Penalized Bellman operator
            backup = rewards + self.gamma * (1 -
                                             done) * (qc_targ <= self.q_thres) * q_targ
        # MSE loss against Bellman backup
        loss_critic = self.critic.loss(backup, q_list)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
        stats_critic = {"loss/critic_loss": loss_critic.item()}
        return loss_critic, stats_critic

    def actor_loss(self, observations):
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.cost_critic.parameters():
            p.requires_grad = False

        actions, _ = self._actor_forward(observations, False, True)
        q_pi, _ = self.critic.predict(observations, actions)
        qc_pi, _ = self.cost_critic.predict(observations, actions)
        loss_actor = -((qc_pi <= self.q_thres) * q_pi).mean()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        stats_actor = {"loss/actor_loss": loss_actor.item()}

        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.cost_critic.parameters():
            p.requires_grad = True
        return loss_actor, stats_actor

    def sync_weight(self):
        """
        Soft-update the weight for the target network.
        """
        self._soft_update(self.critic_old, self.critic, self.tau)
        self._soft_update(self.cost_critic_old, self.cost_critic, self.tau)
        self._soft_update(self.actor_old, self.actor, self.tau)

    def setup_optimizers(self, actor_lr, critic_lr, alpha_lr, vae_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.cost_critic_optim = torch.optim.Adam(self.cost_critic.parameters(),
                                                  lr=critic_lr)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)
        self.alpha_lr = alpha_lr
