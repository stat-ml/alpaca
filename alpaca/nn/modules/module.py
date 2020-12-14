from typing import Optional
import torch.nn as nn


class Module:
    """
    The class links nn.Module with the alpaca Module abstraction
    by allowing us to copy nn.Module instance's dictionary into
    this class instance. Additionally, the class introduces
    additional flags for the inference/uncertainty estimation modes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_mode = False

    def instantiate_with_dropout_params(
        self,
        module: nn.Module,
        dropout_rate: float = 0.0,
        dropout_mask: "BaseMask" = None,
    ) -> "alpaca.nn.Module":
        """
        Copies the instant nn.Module but also adding
        dropout_mask/dropout_rate parameters

        Parameters
        ----------

        module : nn.Module
            The instance nn.Module to be copied
        dropout_rate : float
            The dropout rate
        dropout_mask : "BaseMask"
            Base mask instance setting the type of mask of the module
        """
        self.__dict__ = module.__dict__.copy()
        self.dropout_rate = dropout_rate
        self.dropout_mask = dropout_mask
        return self

    def __str__(self) -> str:
        return "ann.{}, dropout_rate: {}, dropout_mask: {}".format(
            self.__class__.__name__,
            self.dropout_rate,
            self.dropout_mask.__class__.__name__,
        )

    def ue_mode(self) -> "alpaca.nn.Module":
        """
        Sets the alpaca.Module into the uncertainty estimaton mode.
        This will enable the dropout mask logic calculation with the
        dropout rate activated.
        """
        self.uncertainty_mode = True
        return self

    def inf_mode(self) -> "alpaca.nn.Module":
        """
        Sets the alpaca.Module into inference mode. This will disable
        dropout_rate and dropout_mask of the module.
        """
        self.uncertainty_mode = False
        return self
