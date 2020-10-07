import torch

__all__ = ["corrcoef"]


def corrcoef(x: torch.Tensor) -> torch.Tensor:
    """
    TODO: ... np.corrcoef

    Parameters
    ----------
    x : torch.Tensor
        ...
    Returns
    -------
    torch.Tensor
    """
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x[..., None].expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c


def mc_probability(prob: torch.Tensor, k: int, repeats: int = 1000) -> torch.Tensor:
    """
    TODO: docs

    Parameters
    ----------
    prob : torch.Tensor
        ...
    k : int
        ...
    repeats : int
        ...

    Returns
    -------
    torch.Tensor
    """
    results = torch.zeros((repeats, len(prob)))
    for i in range(repeats):
        inds = torch.multinomial(prob, k, replacement=False)
        results[i, inds] = 1.0
    return results.sum(axis=0).div_(repeats)


def cov(m: torch.Tensor, rowvar: bool = True, inplace: bool = False):
    """
    TODO:
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
