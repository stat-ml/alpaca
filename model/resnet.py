import torch
from torch import nn
import torch.nn.functional as F

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, BasicBlock, conv3x3, conv1x1, Bottleneck
# from torchvision.models.resnet import resnet18


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


class ResNetMasked(ResNet):
    def forward(self, x, dropout_rate=0.5, dropout_mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if dropout_mask is not None:
            x = x * dropout_mask(x, dropout_rate, 0)
        x = self.fc(x)

        return x


class ResNetLinear(ResNet):
    def forward(self, x, dropout_rate=0.5, dropout_mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.celu(self.fc(x))
        if dropout_mask is not None:
            x = x * dropout_mask(x, dropout_rate, 0)
        else:
            x = F.dropout(x, self.dropout_rate)
        x = F.celu(self.fc2(x))
        x = F.dropout(x, self.dropout_rate)
        x = self.fc3(x)

        return x


def resnet_linear(pretrained=False, dropout_rate=0.5, freeze=False):
    base = resnet18(pretrained=pretrained, mode='linear')
    base.dropout_rate = dropout_rate
    base.fc = nn.Linear(512, 512)
    base.fc2 = nn.Linear(512, 256)
    base.fc3 = nn.Linear(256, 10)

    if freeze:
        for param in list(base.parameters())[:-8]:
            param.requires_grad = False

    return base


def resnet_masked(pretrained=False):
    base = resnet18(pretrained=pretrained)
    base.fc = nn.Linear(512, 10)

    return base


def _resnet(arch, block, layers, pretrained, progress, mode='masked', **kwargs):
    if mode == 'masked':
        model = ResNetMasked(block, layers, **kwargs)
    else:
        model = ResNetLinear(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

