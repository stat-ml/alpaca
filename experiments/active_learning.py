import torch
from fastai.vision import (
    untar_data, rand_pad, flip_lr, ImageDataBunch, Learner, accuracy,
    cifar_stats, URLs, simple_cnn)
from fastai.vision.models.wrn import wrn_22
from fastai.vision.models import resnet18, resnet34, resnet50

torch.cuda.set_device(1)

torch.backends.cudnn.benchmark = True
path = untar_data(URLs.CIFAR)

ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=512).normalize(cifar_stats)


# model = simple_cnn((3, 256, 256, 128, 10))
# model = wrn_22()
# model = resnet18()
model = resnet50()
learn = Learner(data, model, metrics=accuracy).to_fp16()
learn.fit_one_cycle(30, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)

