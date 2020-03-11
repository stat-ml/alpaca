from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


def scale(train, val):
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    val = scaler.transform(val)
    return train, val, scaler


# Set initial datas
def loader(x, y, batch_size=128, task='regression'):
    if task == 'regression':
        ds = TensorDataset(torch.DoubleTensor(x), torch.DoubleTensor(y))
    elif task == 'classification':
        ds = TensorDataset(torch.DoubleTensor(x), torch.LongTensor(y))
    _loader = DataLoader(ds, batch_size=batch_size)
    return _loader


