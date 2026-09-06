import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

# --- 1. The Dataset Class ---
class MDTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, args, repeat=1):
        super().__init__()
        self.data_file = args.data_file
        self.args = args
        self.repeat = repeat
        
        # Box dimensions for unwrapping coordinates
        if hasattr(self.args, 'box') and self.args.box is not None:
            self.Lx, self.Ly, self.Lz = self.args.box
        else:
            self.Lx, self.Ly, self.Lz = None, None, None

        # Check for large_file flag, defaulting to True (memmap) for safety
        self.large_file = getattr(self.args, 'large_file', True)

        if not self.large_file:
            # Load the entire dataset into RAM
            self.data = np.load(self.data_file)
            if hasattr(self.args, 'frame_interval') and self.args.frame_interval > 1:
                self.data = self.data[::self.args.frame_interval]
                
            self.total_frames = self.data.shape[0]
            self.num_atoms = self.data.shape[1]
            self.worker_arr = None
        else:
            # Probe the memmap ONCE in the main process to get dimensions
            temp_arr = np.lib.format.open_memmap(self.data_file, mode='r')
            if hasattr(self.args, 'frame_interval') and self.args.frame_interval > 1:
                self.total_frames = len(temp_arr[::self.args.frame_interval])
            else:
                self.total_frames = temp_arr.shape[0]
                
            self.num_atoms = temp_arr.shape[1]
            del temp_arr # Close the file immediately
            
            # This will hold the file descriptor for each individual worker
            self.worker_arr = None

    def __len__(self):
        return self.repeat

    def __getitem__(self, idx):
        # Select target time range efficiently
        max_start = self.total_frames - self.args.num_frames
        frame_start = 1 if getattr(self.args, 'overfit_frame', False) else np.random.randint(0, max_start)
        end = frame_start + self.args.num_frames
        
        if not self.large_file:
            # Slice directly from RAM
            chunk = np.copy(self.data[frame_start:end]).astype(np.float32)
        else:
            # LAZY LOAD: Each worker opens the file once and keeps it open
            if self.worker_arr is None:
                arr = np.lib.format.open_memmap(self.data_file, mode='r')
                if hasattr(self.args, 'frame_interval') and self.args.frame_interval > 1:
                    self.worker_arr = arr[::self.args.frame_interval]
                else:
                    self.worker_arr = arr
            
            # Slice from the open worker array
            chunk = np.copy(self.worker_arr[frame_start:end]).astype(np.float32)
        
        if hasattr(self.args, 'copy_frames') and self.args.copy_frames:
            chunk[1:] = chunk[0]

        x, y, z = chunk[..., 4], chunk[..., 5], chunk[..., 6]
        
        trans = np.stack([x, y, z], axis=-1)

        # Extract quaternions (rots)
        qw, qx, qy, qz = chunk[..., 0], chunk[..., 1], chunk[..., 2], chunk[..., 3]
        rots = np.stack([qw, qx, qy, qz], axis=-1)

        # Convert to PyTorch tensors
        trans_t = torch.from_numpy(trans).float()
        rots_t = torch.from_numpy(rots).float()
        
        mask = torch.ones(trans_t.shape[:-1], dtype=torch.float32)

        return {
            'frame_start': frame_start,
            'trans': trans_t,
            'rots': rots_t,
            'mask': mask
        }