import pandas as pd
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, args, metadata, challenge_data=None):
        self.metadata = metadata
        self.args = args
        self.challenge_data = challenge_data

    def __len__(self):
        if self.challenge_data is None:
            return len(self.metadata)
        else:
            return len(self.challenge_data)

    def __getitem__(self, idx):
        """
        Dynamically loads the required chunk of data based on metadata.
        """
        if self.challenge_data is None:
            item = self.metadata[idx]
            file_path, start_idx, end_idx = (
                item["file_path"],
                item["start_idx"],
                item["end_idx"],
            )
            chunk = pd.read_csv(
                file_path, skiprows=range(1, start_idx), nrows=end_idx - start_idx
            )
            # Extract agent positions and velocities (x, y, vx, vy)
            data = []
        else:
            chunk = self.challenge_data[idx]
            data = []

        for agent in self.args.agents:
            agent_data = chunk[
                [f"{agent}_x", f"{agent}_y", f"{agent}_vx", f"{agent}_vy"]
            ].values
            if self.challenge_data is not None:
                agent_data = agent_data[
                    -self.args.burn_in :
                ]  # should be modified later
            data.append(agent_data)

        # Stack agents and convert to tensor
        tensor = torch.tensor(data, dtype=torch.float32)  # Shape: (agents, length, dim)

        if self.args.Modify_Velocity:
            vel = (tensor[:, 1:, 0:2] - tensor[:, :-1, 0:2]) * self.args.fs
            tensor[:, :-1, 2:4] = vel

        tensor = tensor.permute(1, 0, 2)  # Shape: (length, agents, dim)
        tensor = tensor.reshape(tensor.size(0), -1)  # Flatten the last two dimensions
        tensor = tensor.unsqueeze(0)  # agents, time, dim
        return tensor
