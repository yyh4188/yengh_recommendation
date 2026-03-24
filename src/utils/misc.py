import torch
import numpy as np
import matplotlib.pyplot as plt


def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


def write_losses(writer, loss_dict, kind="train"):
    def write_loss(kind, key, item, step):
        writer.add_scalar(kind + "/" + key, item, global_step=step)

    step = loss_dict["step"]
    for k, v in loss_dict.items():
        if k == "step":
            continue
        write_loss(kind, k, v, step)


def pairwise_distances_fig(x):
    # x is (batch_size, action_dim)
    # returns fig
    x = x.cpu().detach().numpy()
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(x))
    fig, ax = plt.subplots()
    im = ax.imshow(dists)
    plt.colorbar(im)
    return fig


class DummyWriter:
    def add_figure(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass
