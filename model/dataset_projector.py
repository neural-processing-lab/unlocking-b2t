import torch

class DatasetPadding(torch.nn.Module):
    """A simple padding module.

    This module pads the input tensor to a specified sensor dimension. If the input tensor is larger than the
    sensor dimension, it raises an error.
    """

    def __init__(self, sensor_dim):
        super().__init__()
        self.sensor_dim = sensor_dim
    
    def forward(self, x):
        if x.shape[1] < self.sensor_dim:
            # Pad the input tensor to the sensor dimension
            x = torch.nn.functional.pad(x, (0, 0, 0, self.sensor_dim - x.shape[1]))
        elif x.shape[1] > self.sensor_dim:
            raise ValueError("Input tensor is larger than the sensor dimension.")
        
        return x

class DatasetProjector(torch.nn.Module):
    """A convolutional projector.

    This module takes tensors from multiple datasets (padded to the same sensor dimension) and projects them to a
    shared space. It uses dataset-conditional adapters to account for dataset differences.
    """

    def __init__(self, sensor_dim, model_dim, num_datasets):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.num_datasets = num_datasets
        self.projector = self._build_projector(sensor_dim, num_datasets * model_dim)
        self.dataset_padding = DatasetPadding(sensor_dim)

    def _build_projector(self, sensor_dim, model_dim):
        # Purely spatial dataset projection
        conv1 = torch.nn.Sequential(
            # Spatial and mildy temporal convolution that projects the input tensor to the model dimension
            torch.nn.Conv1d(sensor_dim, model_dim, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )
        return conv1

    def forward(self, x, dataset_id):
        dataset_onehot = torch.nn.functional.one_hot(
            dataset_id, num_classes=self.num_datasets
        )  # [B, M]

        x = self.dataset_padding(x)  # Pad sensor dimension

        # Project (using all weights in conv)
        x = self.projector(x)  # [B, M * C, T]

        # Apply gating using the one-hot dataset embedding to select only the output computed from the weights for the
        # correct respective dataset (i.e. computes a dataset-specific convolution).
        x = torch.unflatten(x, 1, (self.num_datasets, -1))  # [B, M, C, T]
        x = (x * dataset_onehot[:, :, None, None]).sum(dim=1)  # [B, C, T]

        return x

class SharedProjector(torch.nn.Module):
    """A shared projector.

    This module takes tensors from multiple datasets (padded to the same sensor dimension) and projects them to a
    shared space. It uses dataset-conditional adapters to account for dataset differences.
    """

    def __init__(self, sensor_dim, model_dim):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.projector = self._build_projector(sensor_dim, model_dim)
        self.dataset_padding = DatasetPadding(sensor_dim)

    def _build_projector(self, sensor_dim, model_dim):
        # Purely spatial dataset projection
        conv1 = torch.nn.Sequential(
            # Spatial and mildy temporal convolution that projects the input tensor to the model dimension
            torch.nn.Conv1d(sensor_dim, model_dim, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )
        return conv1

    def forward(self, x):
        x = self.dataset_padding(x)  # Pad sensor dimension

        # Project (using all weights in conv)
        x = self.projector(x)  # [B, M, T]

        return x