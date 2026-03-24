import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import Actor, Critic
from src.algorithms import ddpg_update
from src.utils import soft_update, DummyWriter
import copy


def simple_example():
    """
    简单的DDPG示例，不依赖外部数据
    """
    print("Running simple DDPG example...")

    device = torch.device('cpu')

    input_dim = 64
    action_dim = 32
    hidden_size = 128
    batch_size = 8

    print(f"Input dim: {input_dim}, Action dim: {action_dim}, Hidden size: {hidden_size}")

    policy_net = Actor(input_dim, action_dim, hidden_size).to(device)
    value_net = Critic(input_dim, action_dim, hidden_size).to(device)

    target_policy_net = copy.deepcopy(policy_net)
    target_value_net = copy.deepcopy(value_net)

    target_policy_net.eval()
    target_value_net.eval()

    soft_update(value_net, target_value_net, soft_tau=1.0)
    soft_update(policy_net, target_policy_net, soft_tau=1.0)

    params = {
        'gamma': 0.99,
        'min_value': -10,
        'max_value': 10,
        'policy_step': 5,
        'soft_tau': 0.001,
    }

    nets = {
        'value_net': value_net,
        'target_value_net': target_value_net,
        'policy_net': policy_net,
        'target_policy_net': target_policy_net,
    }

    optimizers = {
        'policy_optimizer': torch.optim.Adam(policy_net.parameters(), lr=1e-4),
        'value_optimizer': torch.optim.Adam(value_net.parameters(), lr=1e-4),
    }

    print("Training for 20 steps...")
    for step in range(20):
        state = torch.randn(batch_size, input_dim).to(device)
        action = torch.randn(batch_size, action_dim).to(device)
        reward = torch.randn(batch_size, 1).to(device)
        next_state = torch.randn(batch_size, input_dim).to(device)
        done = torch.zeros(batch_size, 1).to(device)

        batch = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        losses = ddpg_update(
            batch,
            params,
            nets,
            optimizers,
            device=device,
            writer=DummyWriter(),
            learn=True,
            step=step
        )

        if step % 5 == 0:
            print(f"Step {step}: Value Loss = {losses['value']:.4f}, Policy Loss = {losses['policy']:.4f}")

    print("Training completed!")

    test_state = torch.randn(1, input_dim).to(device)
    test_action = policy_net(test_state)
    test_value = value_net(test_state, test_action)

    print(f"\nTest action shape: {test_action.shape}")
    print(f"Test value: {test_value.item():.4f}")
    print("Example completed successfully!")


if __name__ == "__main__":
    simple_example()
