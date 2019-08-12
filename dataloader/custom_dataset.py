class CustomDataset:
    def __init__(self, x_set, y_set):
        self.x_set = x_set
        self.y_set = y_set

    def __getitem__(self, index):
        return self.x_set[index], self.y_set[index]

    def __len__(self):
        return len(self.x_set)

