import torch
from torch.utils.data import Dataset


class StrokeDataset(Dataset):
    def __init__(self, file_path):
        # Load the dictionary from the .pt file
        self.data_dict = torch.load(file_path)
        self.keys = list(
            self.data_dict.keys()
        )  # List of keys (e.g., 'pat_0', 'pat_1', ...)

    def __len__(self):
        # Return the number of patients (or slices)
        return len(self.keys)

    def normalize(self, data):
        # Compute mean and std for each channel
        mean = data.mean(dim=[0, 2, 3], keepdim=True)  # Mean across N, H, W
        std = data.std(dim=[0, 2, 3], keepdim=True)  # Std across N, H, W
        # Normalize the tensor
        normalized_data = (data - mean) / (
            std + 1e-8
        )  # Add small epsilon to avoid division by zero
        return normalized_data

    def __getitem__(self, idx):
        # Retrieve the key for the given index
        key = self.keys[idx]
        # Load the corresponding tensor for that patient
        tensor = self.data_dict[key]

        tensor[0] = (tensor[0] - torch.mean(tensor[0])) / (torch.std(tensor[0]) + 1e-5)
        tensor[1] = (tensor[1] - torch.mean(tensor[1])) / (torch.std(tensor[1]) + 1e-5)
        tensor[2] = torch.round(tensor[2])
        print(torch.mean(tensor[0]))
        print(torch.max(tensor[2]))
        return tensor
