import sklearn
import numpy as np

import math
from torch.nn import functional as f
import torch
from torch import nn, optim
import math
from scipy.special import softmax


class Calibrator():
    def __init__(self, logits, labels):
        self.temperature = torch.ones(1, requires_grad=True)
        self.logits = logits
        self.labels = labels
        self.W = torch.diag(torch.ones(logits.shape[1]))
        self.W.requires_grad_()
        self.b = torch.zeros(logits.shape[1], requires_grad=True)
        self.W_diag = torch.cat((torch.ones(logits.shape[1]), torch.zeros(logits.shape[1])), dim=0)
        self.W_diag.requires_grad_()

    def split_into_bins(self, n_bins, logits, labels):
        bins = []
        true_labels_for_bins = []

        for i in range(n_bins):
            bins.append([])
            true_labels_for_bins.append([])

        for j in range(len(labels)):
            max_p = max(softmax(logits[j]))
            for i in range(n_bins):
                if i / n_bins < max_p and max_p <= (i + 1) / n_bins:
                    bins[i].append((logits[j]))
                    true_labels_for_bins[i].append(labels[j])
        return np.array(bins), np.array(true_labels_for_bins)

    def compute_ece(self, n_bins, logits, labels, len_dataset):
        bins, true_labels_for_bins = self.split_into_bins(n_bins, logits, labels)
        bins = list(filter(None, bins))
        true_labels_for_bins = list(filter(None, true_labels_for_bins))
        ece = torch.zeros(1)
        for i in range(len(bins)):
            softmaxes = f.softmax(torch.from_numpy(np.array(bins[i])), dim=1)
            confidences, predictions = torch.max(softmaxes, dim=1)
            accuracy = sklearn.metrics.accuracy_score(true_labels_for_bins[i], predictions)
            confidence = torch.sum(confidences) / len(bins[i])
            ece += len(bins[i]) * torch.abs(accuracy - confidence) / len_dataset
        return ece

    def split_into_classes(self, dataset, column_label, logits):
        by_column = dataset.groupby(column_label)
        datasets = {}
        class_logits = []
        dict_class_logits = {}
        n_classes = len(set(dataset[column_label]))
        for i in range(n_classes):
            class_logits.append([])
        for groups, data in by_column:
            datasets[groups] = data
        for ind, label in enumerate(dataset[column_label].to_numpy()):
            for i in range(n_classes):
                if label == i:
                    class_logits[i].append(logits[ind])
        for i in range(n_classes):
            dict_class_logits[i] = class_logits[i]
        return datasets, dict_class_logits

    def compute_sce(self, nbins, column_label, logits, dataset):
        ece_values_for_each_class = []
        datasets, dict_class_logits = self.split_into_classes(dataset, column_label, logits)
        for item in datasets.keys():
            ece_values_for_each_class.append(
                self.compute_ece(nbins, dict_class_logits[item], datasets[item][column_label].to_numpy(), len(dataset)))
        return sum(ece_values_for_each_class) / len(datasets.keys())

    def SplitIntoRanges(self, R, logits, labels):
        N = len(logits)
        bins = []
        true_labels = []
        for i in range(R):
            bins.append([])
            true_labels.append([])
        for j in range(R):
            for i in range(j * math.floor(N / R), (j + 1) * math.floor(N / R)):
                bins[j].append(logits[i])
                true_labels[j].append(labels[i])
        return np.array(bins), np.array(true_labels)

    def ComputeAce(self, R, dataset, target, logits):
        datasets, dict_class_logits = self.split_into_classes(dataset, target, logits)
        summa = 0
        for dataset in datasets.keys():
            data = datasets[dataset]
            class_labels = data[target].to_numpy()
            class_logits = dict_class_logits[dataset]
            bins, true_labels = self.SplitIntoRanges(R, class_logits, class_labels)
            for binn, bin_labels in zip(bins, true_labels):
                softmaxes = f.softmax(torch.from_numpy(binn), dim=1)
                accuracy = sklearn.metrics.accuracy_score(torch.from_numpy(bin_labels), np.argmax(softmaxes, axis=1))
                conf_array = torch.max(softmaxes, dim=1)[0]
                confidence = torch.sum(conf_array) / len(conf_array)
                substraction = abs(accuracy - confidence)
                summa += substraction
        ACE = summa / (len(datasets.keys()) * R)
        return ACE

    def ChooseData(self, threshold, dataset, logits):
        arr = torch.max(f.softmax(torch.from_numpy(logits), dim=1), dim=1)[0]
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
        chosen_data = dataset.iloc[indices]
        chosen_logits = logits[indices]
        return chosen_data, chosen_logits

    def ComputeTace(self, threshold, dataset, logits, R, target):
        chosen_data, chosen_logits = self.ChooseData(threshold, dataset, logits)
        return self.ComputeAce(R, chosen_data, target, chosen_logits)

    def NumberOfClasses(self, dataset, target):
        by_column = dataset.groupby(target)
        datasets = {}
        for groups, data in by_column:
            datasets[groups] = data
        return len(datasets)

    def matrix_scaling_logits(self, logits):
        self.b.unsqueeze(0).expand(logits.shape[0], -1)
        return torch.mm(torch.from_numpy(logits), self.W) + self.b

    def vector_scaling_logits(self, logits):
        W = torch.diag(self.W_diag[:logits.shape[1]])
        b = self.W_diag[logits.shape[1]:]
        b = b.unsqueeze(0).expand(logits.shape[0], -1)
        return torch.mm(torch.from_numpy(logits), W) + b

    def scale_logits_with_temperature(self, logits):
        self.temperature.unsqueeze(1).expand(logits.shape[0], logits.shape[1])
        return torch.true_divide(torch.from_numpy(logits), self.temperature)

    def TemperatureScaling(self):
        nll = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.0001, max_iter=500)

        def eval():
            loss = nll(self.scale_logits_with_temperature(self.logits), torch.from_numpy(np.array(self.labels)))
            loss.backward()
            return loss

        optimizer.step(eval)
        return self

    def MatrixScaling(self):

        nll = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.W, self.b], lr=0.0001, max_iter=1000)

        def eval():
            loss = nll(self.matrix_scaling_logits(self.logits), torch.from_numpy(np.array(self.labels)))
            loss.backward()
            return loss

        optimizer.step(eval)
        return self

    def VectorScaling(self):
        nll = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.W_diag], lr=0.000001, max_iter=9000)

        def eval():
            loss = nll(self.vector_scaling_logits(self.logits), torch.from_numpy(np.array(self.labels)))
            loss.backward()
            return loss

        optimizer.step(eval)
        return self


def binary_histogram_binning(num_bins, probs, labels, probs_to_calibrate):
    bins = np.linspace(0, 1, num=num_bins)
    indexes_list = np.digitize(probs, bins) - 1
    theta = np.zeros(num_bins)
    for i in range(len(bins)):
        binn = (indexes_list == i)
        binn_len = np.sum(binn)
        if binn_len != 0:
            theta[i] = np.sum(labels[binn]) / binn_len
        else:
            theta[i] = bins[i]
    return list(map(lambda x: theta[np.digitize(x, bins) - 1], probs_to_calibrate))


def multiclass_histogram_binning(num_bins, logits, labels, logits_to_calibrate):
    probs = softmax(logits, axis=1)
    probs_to_calibrate = softmax(logits_to_calibrate, axis=1)
    binning_res = []
    for k in range(np.shape(probs)[1]):
        binning_res.append(binary_histogram_binning(num_bins, probs[:, k], labels == k, probs_to_calibrate[:, k]))
    binning_res = np.vstack(binning_res).T
    cal_confs = binning_res / (np.sum(binning_res, axis=1)[:, None])
    return cal_confs

