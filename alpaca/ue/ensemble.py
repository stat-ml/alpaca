from typing import List
import torch.nn as nn


class Ensemble:
    def __init__(self, models: List[nn.Module], reduction: str = "mean"):
        self.models = models
        self.reduction = reduction
