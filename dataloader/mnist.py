from .openml import Openml


class MnistData(Openml):
    def __init__(self, **kwargs):
        super().__init__('mnist_784', **kwargs)


class FashionMnistData(Openml):
    def __init__(self, **kwargs):
        super().__init__('Fashion-MNIST', **kwargs)

