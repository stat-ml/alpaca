from sklearn.metrics import accuracy_score
import numpy as np
import math
from torch.nn import functional as f
import torch
from torch import nn, optim
from scipy.special import softmax
import pandas as pd


def _split_into_bins(n_bins, probs, labels):
    bins = []
    true_labels_for_bins = []

    for i in range(n_bins):
        bins.append([])
        true_labels_for_bins.append([])

    for j in range(len(labels)):
        max_p = max(probs[j])
        for i in range(n_bins):
            if i / n_bins < max_p and max_p <= (i + 1) / n_bins:
                bins[i].append((probs[j]))
                true_labels_for_bins[i].append(labels[j])
    return np.array(bins), np.array(true_labels_for_bins)


def compute_ece(n_bins, probs, labels, len_dataset):
    bins, true_labels_for_bins = _split_into_bins(n_bins, probs, labels)
    bins = list(filter(None, bins))
    true_labels_for_bins = list(filter(None, true_labels_for_bins))
    ece = torch.zeros(1)
    for i in range(len(bins)):
        softmaxes = torch.from_numpy(np.array(bins[i]))
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracy = accuracy_score(true_labels_for_bins[i], predictions)
        confidence = torch.sum(confidences) / len(bins[i])
        ece += len(bins[i]) * torch.abs(accuracy - confidence) / len_dataset
    return ece


def _split_into_classes(labels, probs):
    class_probs = []
    dict_class_probs = {}
    n_classes = np.shape(probs)[1]
    for i in range(n_classes):
        class_probs.append([])
    for ind, label in enumerate(labels):
        for i in range(n_classes):
            if label == i:
                class_probs[i].append(probs[ind])
    for i in range(n_classes):
        dict_class_probs[i] = class_probs[i]
    return dict_class_probs


def compute_sce(nbins, probs, labels):
    ece_values_for_each_class = []
    dict_class_probs = _split_into_classes(labels, probs)
    for item in dict_class_probs.keys():
        ece_values_for_each_class.append(
            compute_ece(
                nbins,
                dict_class_probs[item],
                np.array([item] * np.shape(dict_class_probs[item])[0]),
                len(labels),
            )
        )
    return sum(ece_values_for_each_class) / len(dict_class_probs.keys())


def _split_into_ranges(R, probs, labels):
    N = len(probs)
    bins = []
    true_labels = []
    for i in range(R):
        bins.append([])
        true_labels.append([])
    for j in range(R):
        for i in range(j * math.floor(N / R), (j + 1) * math.floor(N / R)):
            bins[j].append(probs[i])
            true_labels[j].append(labels[i])
    return np.array(bins), np.array(true_labels)


def compute_ace(R, probs, labels):
    dict_class_probs = _split_into_classes(labels, probs)
    summa = 0
    for item in dict_class_probs.keys():
        class_labels = np.array([item] * np.shape(dict_class_probs[item])[0])
        class_probs = dict_class_probs[item]
        bins, true_labels = _split_into_ranges(R, class_probs, class_labels)
        for binn, bin_labels in zip(bins, true_labels):
            conf_array, predictions = torch.max(torch.from_numpy(binn), dim=1)
            accuracy = accuracy_score(bin_labels, predictions.numpy())
            confidence = torch.sum(conf_array) / len(conf_array)
            substraction = abs(accuracy - confidence)
            summa += substraction
    ACE = summa / (len(dict_class_probs.keys()) * R)
    return ACE


def _choose_data(threshold, probs, labels):
    arr = torch.max(torch.from_numpy(np.array(probs)), dim=1)[0]
    arr.numpy()
    arr_with_indices = list(enumerate(arr))
    arr_with_indices.sort(key=lambda x: x[1])
    thr_array = []
    for pair in arr_with_indices:
        if pair[1] > threshold:
            thr_array.append(pair)
    indices = []
    for pair in thr_array:
        indices.append(pair[0])
    chosen_labels = labels[indices]
    chosen_probs = probs[indices]
    return chosen_labels, chosen_probs


def compute_tace(threshold, probs, labels, R):
    if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
        labels = labels.to_numpy()
    chosen_labels, chosen_probs = _choose_data(threshold, probs, labels)
    return compute_ace(R, chosen_probs, chosen_labels)


class ModelWithTempScaling(nn.Module):
    """
    A wrapper for a model with temperature scaling

    model: a classification neural network
    n_classes: number of classes in the dataset
    """

    def __init__(self, model):
        super(ModelWithTempScaling, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, input):
        logits = self.model(input)
        return f.softmax(torch.true_divide(logits, self.temperature), dim=1)

    def scaling(self, logits, labels, lr=0.01, max_iter=50):
        # logits and labels must be from calibration dataset
        nll = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            loss = nll(torch.true_divide(logits, self.temperature), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        return self


class ModelWithVectScaling(nn.Module):

    """
    A wrapper for a model with vector scaling

    model:  a classification neural network
    n_classes: number of classes in the dataset

    """

    def __init__(self, model, n_classes):
        super(ModelWithVectScaling, self).__init__()
        self.model = model
        self.W_and_b = nn.Parameter(
            torch.cat((torch.ones(n_classes), torch.zeros(n_classes)), dim=0)
        )

    def forward(self, input):
        logits = self.model(input)
        return f.softmax(self.scaling_logits(logits), dim=1)

    def scaling_logits(self, logits):
        # logits and labels must be from calibration dataset
        W = torch.diag(self.W_and_b[: logits.shape[1]])
        b = self.W_and_b[logits.shape[1] :]
        b = b.unsqueeze(0).expand(logits.shape[0], -1)
        return torch.mm(logits.float(), W) + b

    def scaling(self, logits, labels, lr=0.00001, max_iter=3500):
        nll = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.W_and_b], lr=lr, max_iter=max_iter)

        def eval():
            loss = nll(self.scaling_logits(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        return self


class ModelWithMatrScaling(nn.Module):
    """
    A wrapper for a model with matrix scaling

    model: a classification neural network
    n_classes: number of classes in the dataset
    """

    def __init__(self, model, n_classes):
        super(ModelWithMatrScaling, self).__init__()
        self.model = model
        self.W = nn.Parameter(torch.diag(torch.ones(n_classes)))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, input):
        logits = self.model(input)
        return f.softmax(self.scaling_logits(logits), dim=1)

    def scaling_logits(self, logits):
        self.b.unsqueeze(0).expand(logits.shape[0], -1)
        return torch.mm(logits.float(), self.W) + self.b

    def scaling(self, logits, labels, lr=0.001, max_iter=100):
        # logits and labels must be from calibration dataset
        nll = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.W, self.b], lr=lr, max_iter=max_iter)

        def eval():
            loss = nll(self.scaling_logits(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        return self


def binary_histogram_binning(num_bins, probs, labels, probs_to_calibrate):
    """
    histogram binning for  binary classification
    :param num_bins: number of bins
    :param probs: probabilities on calibration dataset
    :param labels: labels of calibration dataset
    :param probs_to_calibrate: initial probabilities on test dataset (which need to be calibrated)
    :return: calibrated probabilities on test dataset
    """
    bins = np.linspace(0, 1, num=num_bins)
    indexes_list = np.digitize(probs, bins) - 1
    theta = np.zeros(num_bins)
    for i in range(len(bins)):
        binn = indexes_list == i
        binn_len = np.sum(binn)
        if binn_len != 0:
            theta[i] = np.sum(labels[binn]) / binn_len
        else:
            theta[i] = bins[i]
    return list(map(lambda x: theta[np.digitize(x, bins) - 1], probs_to_calibrate))


def multiclass_histogram_binning(num_bins, logits, labels, logits_to_calibrate):
    """
    histogram binning for multiclass classification
    :param num_bins: number of bins
    :param logits: logits on calibration dataset
    :param labels: labels on calibration dataset
    :param logits_to_calibrate: initial logits on test dataset (which need to be calibrated)
    :return: calibrated probabilities on test dataset
    """
    probs = softmax(logits, axis=1)
    probs_to_calibrate = softmax(logits_to_calibrate, axis=1)
    binning_res = []
    for k in range(np.shape(probs)[1]):
        binning_res.append(
            binary_histogram_binning(
                num_bins, probs[:, k], labels == k, probs_to_calibrate[:, k]
            )
        )
    binning_res = np.vstack(binning_res).T
    cal_confs = binning_res / (np.sum(binning_res, axis=1)[:, None])
    return cal_confs
