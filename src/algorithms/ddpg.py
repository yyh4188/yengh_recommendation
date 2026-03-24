import torch
from ..utils import soft_update, write_losses, DummyWriter, pairwise_distances_fig
from .misc import value_update
from ..data.utils import get_base_batch


def ddpg_update(
    batch,
    params,
    nets,
    optimizer,
    device=torch.device("cpu"),
    debug=None,
    writer=DummyWriter(),
    learn=False,
    step=-1,
):

    """
    :param batch: batch [state, action, reward, next_state] returned by environment.
    :param params: dict of algorithm parameters.
    :param nets: dict of networks.
    :param optimizer: dict of optimizers
    :param device: torch.device
    :param debug: dictionary where debug data about actions is saved
    :param writer: torch.SummaryWriter
    :param learn: whether to learn on this step (used for testing)
    :param step: integer step for policy update
    :return: loss dictionary

    How parameters should look like::

        params = {
            'gamma'      : 0.99,
            'min_value'  : -10,
            'max_value'  : 10,
            'policy_step': 3,
            'soft_tau'   : 0.001,
            'policy_lr'  : 1e-5,
            'value_lr'   : 1e-5,
            'actor_weight_init': 3e-1,
            'critic_weight_init': 6e-1,
            'contrastive_weight': 0.1,  # 对比学习损失权重
        }
        nets = {
            'value_net': models.Critic,
            'target_value_net': models.Critic,
            'policy_net': models.Actor,
            'target_policy_net': models.Actor,
        }
        optimizer - {
            'policy_optimizer': some optimizer
            'value_optimizer':  some optimizer
        }

    """

    state, action, reward, next_state, user_ids = get_base_batch(batch, device=device, return_user_ids=True)
    
    # 调试信息


    # --------------------------------------------------------#
    # Value Learning

    value_loss = value_update(
        batch,
        params,
        nets,
        optimizer,
        writer=writer,
        device=device,
        debug=debug,
        learn=learn,
        step=step,
    )

    # --------------------------------------------------------#
    # Policy learning

    gen_action = nets["policy_net"](state)
    policy_loss = -nets["value_net"](state, gen_action)

    if not learn:
        if debug is not None:
            debug["gen_action"] = gen_action
        writer.add_histogram("policy_loss", policy_loss, step)
        writer.add_figure("next_action", pairwise_distances_fig(gen_action[:50]), step)
    policy_loss = policy_loss.mean()

    # --------------------------------------------------------#
    # Contrastive learning
    contrastive_loss = torch.tensor(0.0, device=device)
    if 'contrastive_weight' in params and params['contrastive_weight'] > 0:
        contrastive_loss = nets["policy_net"].contrastive_loss(gen_action, user_ids)
        writer.add_histogram("contrastive_loss", contrastive_loss, step)

    # Combine losses
    total_policy_loss = policy_loss
    if contrastive_loss > 0:
        total_policy_loss = policy_loss + params['contrastive_weight'] * contrastive_loss

    if learn and step % params["policy_step"] == 0:
        optimizer["policy_optimizer"].zero_grad()
        total_policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(nets["policy_net"].parameters(), -1, 1)
        optimizer["policy_optimizer"].step()

        soft_update(
            nets["value_net"], nets["target_value_net"], soft_tau=params["soft_tau"]
        )
        soft_update(
            nets["policy_net"], nets["target_policy_net"], soft_tau=params["soft_tau"]
        )

    losses = {"value": value_loss.item(), "policy": policy_loss.item(), "contrastive": contrastive_loss.item(), "step": step}
    write_losses(writer, losses, kind="train" if learn else "test")
    return losses
