from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

import matplotlib.colors as colors
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# from utilities.metric import predictive_entropy
# from utilities.metric import expected_entropy
# from utilities.metric import mutual_information

from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder1 = nn.Linear(784, 400)
        self.encoder2 = nn.Linear(400, 256)
        self.mu = nn.Linear(256, 2)
        self.logvar = nn.Linear(256, 2)

        self.decoder1 = nn.Linear(2, 256)
        self.decoder2 = nn.Linear(256, 400)
        self.decoder3 = nn.Linear(400, 784)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        d1 = self.elu(self.encoder1(x))
        d2 = self.elu(self.encoder2(d1))
        return self.mu(d2), self.logvar(d2)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        d1 = self.elu(self.decoder1(z))
        d2 = self.elu(self.decoder2(d1))
        res = self.sigmoid(self.decoder3(d2))
        return res

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD


def train_autoencoder(epoch, train_loader, log_interval, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data.view(-1, 784), mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)
            ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test_autoencoder(test_loader, model):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(
            recon_batch,
            data.view(-1, 1, 28, 28),
            mu,
            logvar
        ).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def show_encoded_decoded_images(images, figsize=(5, 20), cols=5, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    fig = plt.figure(figsize=figsize)

    if titles is None: titles = ['Image (%d)' % i for i in
                                 range(1, n_images + 1)]
    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        if n == 0:
            plt.title('Original image')
        elif n == 1:
            plt.title('Decoded')
        plt.axis('off')
    plt.show()


def show_decoder_quality(mini_batch, model):
    data = mini_batch.cuda()
    data = Variable(data)
    print(data)
    print(data.shape)
    recon_batch, _, _ = model(data)
    decoded = recon_batch.cpu().data.numpy(
    ).reshape(25, 28, 28)
    to_encode = mini_batch.numpy().reshape(25, 28, 28) #[:, 0, :, :]

    alternated = np.zeros((50, 28, 28))
    alternated[0::2] = to_encode
    alternated[1::2] = decoded

    show_encoded_decoded_images(
        alternated,
        cols=25
    )


def create_latent_space_results_mc(
        mesh_min,
        mesh_max,
        model,
        vae,
        delim=0.1,
        num_inferences=50
):
    latent_X, latent_Y = np.meshgrid(np.arange(mesh_min, mesh_max, delim),
                                     np.arange(mesh_min, mesh_max, delim))
    all_results = []

    row_length = latent_X.shape[0]
    decoded_imgs = []

    for i in tqdm(range(row_length)):
        row_X = latent_X[i]
        row_Y = latent_Y[i]

        t = Variable(torch.Tensor(np.stack([row_X, row_Y]).T).cuda())
        decoded = vae.decode(t).view(-1, 1, 28, 28)
        decoded_imgs.append(decoded.cpu().data.numpy())

        list_ = []

        for j in range(num_inferences):
            list_.append((nn.Softmax(1)(model(decoded))).cpu().data.numpy())

        results = np.stack(list_)

        all_results.append(results)

    p_ents = np.zeros((row_length, row_length))
    e_ents = np.zeros((row_length, row_length))
    m_info = np.zeros((row_length, row_length))

    # for i in range(row_length):
    #     p_ents[i] = predictive_entropy(all_results[i])
    #     e_ents[i] = expected_entropy(all_results[i])
    #     m_info[i] = mutual_information(all_results[i])

    return p_ents, e_ents, m_info, latent_X, latent_Y, decoded_imgs


def create_latent_space_results_is(mesh_min, mesh_max, model, vae, delim=0.1):
    latent_X, latent_Y = np.meshgrid(
        np.arange(mesh_min, mesh_max, delim),
        np.arange(mesh_min, mesh_max, delim)
    )
    all_results = []

    row_length = latent_X.shape[0]
    decoded_imgs = []
    model.eval()

    for i in tqdm(range(row_length)):
        row_X = latent_X[i]
        row_Y = latent_Y[i]

        t = Variable(torch.Tensor(np.stack([row_X, row_Y]).T).cuda())
        decoded = vae.decode(t).view(-1, 1, 28, 28)
        decoded_imgs.append(decoded.cpu().data.numpy())

        all_results.append(
            nn.Softmax(1)(model.forward(decoded)[0])[:, 10].cpu().data.numpy()
        )

    return np.stack(all_results), latent_X, latent_Y, decoded_imgs


def plot_latent_space(
        latent_X,
        latent_Y,
        uncertainty,
        xses,
        yses,
        title,
        alpha_latent=0.8,
        alpha_training=0.05,
        size_latent=80,
        size_training=20,
        norm=True,
        savepath=None
):

    plt.figure(figsize=(15, 15), frameon=False)

    con_yses = yses

    if norm:
        norm_ = colors.LogNorm(vmin=uncertainty.flatten().min(),
                               vmax=uncertainty.flatten().max())
    else:
        norm_ = None

    plt.scatter(
        latent_X.flatten(),
        latent_Y.flatten(),
        s=size_latent,
        alpha=alpha_latent,
        c=uncertainty.flatten(),
        norm=norm_,
        cmap='gray',
        marker='s'

    )

    for i in range(10):
        plt.scatter(
            xses[:, 0][con_yses == i],
            xses[:, 1][con_yses == i],
            s=size_training,
            alpha=alpha_training,
            label=i
        )

    plt.title(title)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    if savepath:
        plt.savefig(savepath, bbox_inches = 'tight', pad_inches = 0)


def show_decoded_latent_space(vae, savepath='images/vae_decoded.png'):
    X2, Y2 = np.meshgrid(
        np.arange(-10, 10, 1),
        np.arange(-10, 10, 1),
    )

    input_2 = np.concatenate([X2.reshape(-1, 1), Y2.reshape(-1, 1)], axis=1)
    inf01 = vae.decode(
        Variable(torch.Tensor(input_2)).cuda()).cpu().data.numpy()
    inf_manifold2 = inf01.reshape(20, 20, 28, 28)
    fig = plt.figure(figsize=(15, 15))

    for i in tqdm(range(20)):
        for j in range(20):
            fig.add_subplot(20, 20, i * 20 + j + 1)
            plt.imshow(inf_manifold2[i, j, :, :])
            plt.axis('off')

    plt.savefig(savepath)
