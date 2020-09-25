import torch


class Ensemble:
    def __init__(self, modules):
        self.modules = modules

    def eval(self):
        [m.eval() for m in self.modules]

    def __call__(self, x, reduction="default"):
        res = torch.stack([m(x) for m in self.models])
        if reduction == "default":
            reduction = self.reduction
        if reduction is None:
            res = res
        elif reduction == "mean":
            res = res.mean(dim=0)
        elif reduction == "nll":
            means = res[:, :, 0]
            sigmas = res[:, :, 1]
            res = torch.stack(
                [
                    means.mean(dim=0),
                    sigmas.mean(dim=0)
                    + (means ** 2).mean(dim=0)
                    - means.mean(dim=0) ** 2,
                ],
                dim=1,
            )
        return res
