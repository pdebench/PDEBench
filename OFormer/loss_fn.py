import torch
import torch.nn as nn
from einops import rearrange


def rel_loss(x, y, p, reduction=True, size_average=False, time_average=False):
    # x, y: [b, c, t, h, w] or [b, c, t, n]
    batch_num = x.shape[0]
    frame_num = x.shape[2]

    if len(x.shape) == 5:
        h = x.shape[3]
        w = x.shape[4]
        n = h*w
    else:
        n = x.shape[-1]
    # x = rearrange(x, 'b c t h w -> (b t h w) c')
    # y = rearrange(y, 'b c t h w -> (b t h w) c')
    num_examples = x.shape[0]
    eps = 1e-6
    diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p, 1)
    y_norms = torch.norm(y.reshape(num_examples, -1), p, 1) + eps

    loss = torch.sum(diff_norms/y_norms)
    if reduction:
        loss = loss / batch_num
        if size_average:
            loss /= n
        if time_average:
            loss /= frame_num

    return loss


def rel_l2norm_loss(x, y):
    #   x, y [b, c, t, n]
    eps = 1e-6
    y_norm = (y**2).mean(dim=-1) + eps
    diff = ((x-y)**2).mean(dim=-1)
    diff = diff / y_norm   # [b, c, t]
    diff = diff.sqrt().mean()
    return diff

def pointwise_rel_l2norm_loss(x, y):
    #   x, y [b, n, c]
    eps = 1e-6
    y_norm = (y**2).mean(dim=-2) + eps
    diff = ((x-y)**2).mean(dim=-2)
    diff = diff / y_norm   # [b, c]
    diff = diff.sqrt().mean()
    return diff
