import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(X_data, y_data, batch_size=128, num_workers=4, shuffle=True):
    # Convert X_data to tensor
    if isinstance(X_data, list):
        if all(isinstance(x, torch.Tensor) for x in X_data):
            # If it's a list of tensors, stack them into a single tensor
            X_data_tensor = torch.stack(X_data)
        else:
            # If it's a list of arrays, convert to tensor
            X_data_tensor = torch.tensor(X_data, dtype=torch.float32)
    else:
        # Direct conversion if it's not a list
        X_data_tensor = torch.tensor(X_data, dtype=torch.float32)

    # Convert y_data to tensor
    y_data_tensor = torch.tensor(y_data, dtype=torch.long)

    # Create TensorDataset
    dataset = TensorDataset(X_data_tensor, y_data_tensor)

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader

