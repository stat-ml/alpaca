from typing import Tuple, Optional, Union, Callable
import torch
from tqdm import tqdm

from alpaca.ue.base import UE
from alpaca.models import Ensemble
from alpaca.ue import acquisitions


class EnsembleMCDUE(UE):
    """
    Estimate uncertainty for samples with Ensemble and MCDUE approach
    """

    _name = "EnsembleMCDUE"
    _default_acquisition = acquisitions.std

    def __init__(
        self, *args, acquisition: Optional[Union[str, Callable]] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._create_model_from_list()

    def estimate(self, X_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mcd_runs = None
        predictions = []

        with torch.no_grad():
            # Some mask needs first run without dropout, i.e. decorrelation mask
            self.net(
                X_pool,
            )

            # Get mcdue estimation
            for nn_run in tqdm(range(self.nn_runs), total=self.nn_runs, desc=self.desc):
                prediction = self.net(
                    X_pool,
                )
                mcd_runs = (
                    prediction.flatten().cpu()[None, ...]
                    if mcd_runs is None
                    else torch.cat(
                        [mcd_runs, prediction.flatten().cpu()[None, ...]], dim=0
                    )
                )
                predictions.append(prediction.cpu())

        predictions = torch.cat([*predictions], dim=0)

        # save `mcf_runs` stats
        if self.keep_runs is True:
            self._mcd_runs = mcd_runs
        return predictions, self._acquisition(mcd_runs)

    def _create_model_from_list(self):
        self.net = Ensemble(self.net)
        self.net.eval()


class EnsembleUE(EnsembleMCDUE):
    """
    Estimate uncertainty for samples with Ensemble approach.
    Propose that ensemble contains nets with one output
    """

    def __init__(self, net):
        super(EnsembleUE, self).__init__(net, nn_runs=1, dropout_rate=0)


class EnsembleNLLUE:
    """
    Estimate uncertainty for samples with Ensemble approach.
    Ensemble must contains nets with two outputs: mean and sigma_squared
    """

    def __init__(self, net):
        self.net = net

    def estimate(self, X_pool, **kwargs):
        with torch.no_grad():
            res = self.net(X_pool, reduction="nll")
            sigma = res[:, 1].to("cpu")
            sigma = torch.sqrt(sigma)

        return torch.flatten(sigma)
