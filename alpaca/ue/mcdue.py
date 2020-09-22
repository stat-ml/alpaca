from typing import Tuple
import torch
from tqdm import tqdm

from alpaca.ue.base import UE
from alpaca.ue import acquisitions

__all__ = ["MCDUE"]


class MCDUE:
    """
    MCDUE constructor. Depending on the provided `num_classes` argument, the
    constructor will initialize `MCDUE_regression` or `MCDUE_classification` classes.

    Other Parameters
    ----------------
    num_classes : int
        Integer that sets the number of classes for prediction

    Example
    -------
    MCDUE for classification task with 10 classes
    >>> estimator = MCDUE(model, nn_runs=100, num_classes=10)
    MCDUE for regression task
    >>> estimator = MCDUE(model, nn_runs=100)
    """

    _name = "MCDUE"
    _default_acquisition = None

    def __new__(cls, *args, num_classes: int = 0, **kwargs):
        if num_classes == 0:
            return MCDUE_regression(*args, **kwargs)
        elif num_classes > 0:
            return MCDUE_classification(*args, num_classes=num_classes, **kwargs)
        else:
            raise ValueError("`num_classes` can't take the negative value")


class MCDUE_regression(UE):
    """
    MCDUE implementation for regression task

    Default attributes
    ------------------
    _name : "MCDUE_regression"
    _default_acquisition : :method:`alpaca.ue.acquisitions.std`
    """

    _name = "MCDUE_regression"
    _default_acquisition = acquisitions.std

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def estimate(self, X_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mcd_runs = None
        predictions = []
        with torch.no_grad():
            # Some mask needs first run without dropout, i.e. decorrelation mask
            """
            # TODO:  removed dry run
            if self.dropout_mask:
                self.dropout_mask.dry_run(self.net, X_pool, self.dropout_rate)

            """
            # Get mcdue estimation
            for nn_run in tqdm(range(self.nn_runs), total=self.nn_runs, desc=self.desc):
                prediction = self.net(
                    X_pool,
                    dropout_mask=self.dropout_mask,
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


class MCDUE_classification(UE):
    """
    MCDUE implementation for a classification task

    Default attributes
    ------------------
    _name : "MCDUE_classification"
    _default_acquisition : :method:`alpaca.ue.acquisitions.bald`

    Parameters
    ----------
    num_classes: int
        Integer that sets the number of classes for prediction
    """

    _name = "MCDUE_classification"
    _default_acquisition = acquisitions.bald

    def __init__(self, *args, num_classes: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    def estimate(self, X_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mcd_runs = None
        predictions = []
        with torch.no_grad():
            """
            TODO: remove dry run
            if self.dropout_mask:
                self.dropout_mask.dry_run(self.net, X_pool, self.dropout_rate)
            """

            # Get mcdue estimation
            for nn_run in tqdm(range(self.nn_runs), total=self.nn_runs, desc=self.desc):
                prediction = self.net(
                    X_pool,
                    dropout_mask=self.dropout_mask,
                )
                mcd_runs = (
                    prediction.cpu()[None, ...]
                    if mcd_runs is None
                    else torch.cat([mcd_runs, prediction.cpu()[None, ...]], dim=0)
                )
                predictions.append(prediction.cpu())

        predictions = torch.cat([*predictions], dim=0)
        mcd_runs = mcd_runs.permute((1, 0, 2))

        if self.keep_runs is True:
            self._mcd_runs = mcd_runs

        return predictions, self._acquisition(mcd_runs)
