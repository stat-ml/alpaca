import torch
import torch.utils.data


def loader(x, y, batch_size=64, shuffle=False):
    custom_dataset = CustomDataset(x, y)

    loader = torch.utils.data.DataLoader(
        dataset=custom_dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x_set, y_set):
        self.x_set = torch.from_numpy(x_set)
        self.y_set = torch.from_numpy(y_set)

    def __getitem__(self, index):
        return self.x_set[index], self.y_set[index]

    def __len__(self):
        return len(self.x_set)

