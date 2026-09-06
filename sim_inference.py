import argparse
import os
import time
import torch
import numpy as np
import tqdm

# Import your custom wrapper
from model.wrapper import NewMDGenWrapper
from model.tensor_utils import tensor_tree_map

def get_args():
    parser = argparse.ArgumentParser(description="Inference script for Ellipsoid MDGen")
    parser.add_argument('--sim_ckpt', type=str, required=True, help="Path to the trained .ckpt file")
    parser.add_argument('--data_file', type=str, default="model\\data\\pmma_trajectory_1w.npy", help="Path to the reference .npy data file")
    parser.add_argument('--start_frame', type=int, default=50000, help="Which frame from the data to use as the starting point")
    parser.add_argument('--num_frames', type=int, default=50, help="Number of frames the model generates per rollout (must match training T)")
    parser.add_argument('--num_rollouts', type=int, default=100, help="How many times to autoregressively chain the generation")
    parser.add_argument('--out_file', type=str, default="test/generated_trajectory.npy", help="Path to save the generated .npy trajectory")
    return parser.parse_args()

def get_initial_state(data_file, start_frame):
    """
    Loads a single starting frame from the raw data and unwraps the coordinates.
    """
    arr = np.lib.format.open_memmap(data_file, mode='r')
    
    # Extract just the starting frame: shape (1, N, 11)
    chunk = np.copy(arr[start_frame:start_frame+1]).astype(np.float32)
    
    # Safely get the exact number of particles (N)
    N = chunk.shape[1] 
    
    x, y, z = chunk[..., 4], chunk[..., 5], chunk[..., 6]
        
    trans = np.stack([x, y, z], axis=-1)

    # Extract quaternions (rots)
    qw, qx, qy, qz = chunk[..., 0], chunk[..., 1], chunk[..., 2], chunk[..., 3]
    rots = np.stack([qw, qx, qy, qz], axis=-1)

    # Convert to Tensors and add the Time dimension -> (1, 1, N, D)
    trans_t = torch.from_numpy(trans).float().unsqueeze(1) 
    rots_t = torch.from_numpy(rots).float().unsqueeze(1)   
    
    # FORCE the mask to exactly match the particle count -> (1, N)
    mask_t = torch.ones(1, N, dtype=torch.float32) 

    return {
        'trans': trans_t,
        'rots': rots_t,
        'mask': mask_t
    }

def rollout(model, batch, num_frames):
    """
    Takes a 1-frame batch, expands it to T frames to satisfy model dimensions, 
    runs inference, and extracts the last frame to seed the next rollout.
    """
    # Expand the single starting frame to shape (B=1, T, N, D)
    expanded_batch = {
        'trans': batch['trans'].expand(-1, num_frames, -1, -1),
        'rots': batch['rots'].expand(-1, num_frames, -1, -1),
        'mask': batch['mask'].unsqueeze(1).expand(-1, num_frames, -1)
    } # absolute coordinates in global frame
    
    # Run ODE solver to get absolute trajectories
    # Returns shape: (B=1, T, N, 7) [quats(4), trans(3)]
    generated_traj = model.inference(expanded_batch)
    
    # Extract the very last frame to act as the seed for the next iteration
    # Keep the time dimension so shape remains (1, 1, N, 7)
    last_frame = generated_traj[:, -1:]
    
    new_batch = {
        'rots': last_frame[..., :4],
        'trans': last_frame[..., 4:],
        'mask': batch['mask']
    }
    
    return generated_traj, new_batch

@torch.no_grad() # Disable gradient tracking to save massive amounts of VRAM
def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model from {args.sim_ckpt}...")
    model = NewMDGenWrapper.load_from_checkpoint(args.sim_ckpt)
    model.eval().to(device)
    
    print(f"Loading initial state from frame {args.start_frame}...")
    batch = get_initial_state(args.data_file, args.start_frame) # absolute coordinates in global frame
    
    # Move initial batch to GPU
    batch = tensor_tree_map(lambda x: x.to(device), batch)
    
    all_generated_frames = []
    
    print(f"Starting {args.num_rollouts} autoregressive rollouts...")
    start_time = time.time()
    
    for i in tqdm.trange(args.num_rollouts):
        traj_chunk, batch = rollout(model, batch, args.num_frames)
        
        # To avoid saving overlapping frames, we optionally drop the T=0 frame
        # because T=0 of this chunk is identical to the last frame of the previous chunk.
        if i > 0:
            traj_chunk = traj_chunk[:, 1:]
            
        all_generated_frames.append(traj_chunk.cpu())
    
    print(f"Inference completed in {time.time() - start_time:.2f} seconds.")
    
    # Concatenate all chunks along the time dimension (dim=1)
    # Shape becomes (1, Total_Frames, N, 7)
    full_trajectory = torch.cat(all_generated_frames, dim=1)
    
    # Squeeze out the batch dimension and convert to numpy
    final_numpy_array = full_trajectory.squeeze(0).numpy()
    print(f"Final generated trajectory shape: {final_numpy_array.shape}")
    
    # Save the output
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)) or ".", exist_ok=True)
    np.save(args.out_file, final_numpy_array)
    print(f"Trajectory saved to {args.out_file}")

if __name__ == '__main__':
    main()

# python sim_inference.py --sim_ckpt workdir/ellipsoid_symmetric/epoch=60-step=7625.ckpt --start_frame 1 --num_rollouts 1 --out_file result/generated_trajectory1.npy
# sim_inference --sim_ckpt result\1.2_50frames_PMMA_1w\ellipsoid_symmetric\epoch=98-step=99000.ckpt --start_frame 5000 --num_rollouts 1 --out_file result\1.2_50frames_PMMA_1w\inference 