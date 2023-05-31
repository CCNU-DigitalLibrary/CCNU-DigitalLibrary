import torch
import torch.nn.functional as F


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * x @ y.t()
    # dist.addmm_(x, y.t(), beta=1, alpha=-2)
    # b_dist = dist.addmm(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    assert x.size(0) == y.size(0)
    # compatible with fp16
    x = x.float()
    y = y.float()
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 1 - (x * y).sum(dim=1)
    return dist