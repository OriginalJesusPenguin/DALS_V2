import torch


def sample_in_ball(n):
    """Sample n points uniformly in ball with rejection sampling."""
    def sample(n):
        x = 2 * torch.rand(n, 3) - 1
        keep_idx = torch.sum(x ** 2, dim=1) <= 1
        return x[keep_idx]

    out = sample(2 * n)
    while len(out) < n:
        n_new = max(2 * (len(out) - n), 1000)
        out = torch.cat([out, sample(n_new)])

    return out[:n]

