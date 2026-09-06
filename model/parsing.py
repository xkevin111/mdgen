from argparse import ArgumentParser
import os

def parse_train_args():
    parser = ArgumentParser(description="Train MDGen for Uniaxial Ellipsoid Particles") 

    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a checkpoint to resume training or run inference.") 
    parser.add_argument("--validate", action='store_true', default=False, help="Run validation only (no training).") 
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader worker processes.") 
    
    ## Epoch settings
    group = parser.add_argument_group("Epoch settings") 
    group.add_argument("--epochs", type=int, default=100, help="Total number of training epochs.")
    group.add_argument("--overfit", action='store_true', help="Use the training set as the validation set to test overfitting.")
    group.add_argument("--overfit_frame", action='store_true', help="Force the dataset to always start at frame 0.") 
    group.add_argument("--train_batches", type=int, default=None, help="Number of batches per training epoch.")
    group.add_argument("--val_batches", type=int, default=None, help="Number of batches per validation epoch.")
    group.add_argument("--val_repeat", type=int, default=1, help="Number of times to repeat the validation dataset.")
    group.add_argument("--batch_size", type=int, default=2, help="Number of samples per batch.")
    group.add_argument("--val_freq", type=float, default=1.0, help="Frequency of validation checks within an epoch (fraction or steps).")
    group.add_argument("--val_epoch_freq", type=int, default=1, help="Number of epochs between validation runs.")
    group.add_argument("--no_validate", action='store_true', help="Disable validation entirely.")

    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100, help="Logging frequency in steps.")
    group.add_argument("--ckpt_freq", type=int, default=1, help="Checkpoint saving frequency in epochs.")
    group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    group.add_argument("--run_name", type=str, default="ellipsoid_default", help="Name of the run (used for logging and saving).")
    
    ## Optimization settings
    group = parser.add_argument_group("Optimization settings")
    group.add_argument("--accumulate_grad", type=int, default=4, help="Number of gradient accumulation steps to compensate for small batch sizes.")
    group.add_argument("--grad_clip", type=float, default=1., help="Gradient clipping max norm.")
    group.add_argument("--check_grad", action='store_true', help="Enable gradient checking/debugging.")
    group.add_argument('--grad_checkpointing', action='store_true', help="Enable gradient checkpointing to save VRAM.")
    group.add_argument('--adamW', action='store_true', help="Use AdamW optimizer instead of standard Adam.")
    group.add_argument('--ema', action='store_true', help="Enable Exponential Moving Average (EMA) for model weights.")
    group.add_argument('--ema_decay', type=float, default=0.999, help="Decay rate for EMA.")
    group.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    group.add_argument('--precision', type=str, default='bf16-mixed', help="Mixed precision training setting (e.g., bf16-mixed for RTX 40-series).")
    
    ## Training data 
    group = parser.add_argument_group("Training data settings")
    parser.add_argument('--data_file', type=str, default=r"D:\OneDrive\OneDrive2\OneDrive\ML_project\conference_papers\PMMA_mdgen\data\pmma_trajectory_1w.npy", help="Path to the .npy trajectory data file.")
    group.add_argument('--box', type=float, nargs=3, default=None, help="Simulation box dimensions: Lx Ly Lz")
    group.add_argument('--large_file', action='store_true', help="Use memory-mapped loading for large datasets that don't fit in RAM.")
    group.add_argument('--num_frames', type=int, default=50, help="Number of consecutive frames extracted per sequence.")
    group.add_argument('--crop', type=int, default=330, help="Number of particles/atoms (L) to crop or use in the system.")
    group.add_argument('--frame_interval', type=int, default=1, help="Step size between extracted frames.")
    group.add_argument('--copy_frames', action='store_true', help="Duplicate the first frame across the entire sequence (for debugging).")
    
    ## Model settings
    group = parser.add_argument_group("Model settings")
    group.add_argument('--hyena', action='store_true', help="Use HyenaOperator instead of standard attention for the time dimension.")
    group.add_argument('--use_rope_t', action='store_true', default=True, help="Enable RoPE for the time dimension (T).")
    group.add_argument('--use_rope_l', action='store_true', default=False, help="Enable RoPE for the length/particle dimension (L).")
    group.add_argument('--dropout', type=float, default=0.1, help="Dropout probability.")
    group.add_argument('--interleave_ipa', action='store_true', help="Interleave IPA layers with standard attention blocks.")
    group.add_argument('--prepend_ipa', action='store_true', help="Add IPA layers before the main attention blocks.")
    group.add_argument('--num_layers', type=int, default=4, help="Number of attention layers in the network.")
    group.add_argument('--embed_dim', type=int, default=256, help="Hidden embedding dimension size.")
    group.add_argument('--mha_heads', type=int, default=8, help="Number of heads for standard Multi-Head Attention.")
    group.add_argument('--ipa_heads', type=int, default=4, help="Number of heads for Invariant Point Attention.")
    group.add_argument('--ipa_head_dim', type=int, default=16, help="Hidden dimension size per IPA head.")
    group.add_argument('--ipa_qk', type=int, default=4, help="Number of query/key points generated for IPA.")
    group.add_argument('--ipa_v', type=int, default=4, help="Number of value points generated for IPA.")
    group.add_argument('--time_multiplier', type=float, default=100., help="Scaling factor applied to the time embeddings.")
    group.add_argument('--abs_pos_emb', action='store_true', help="Enable absolute 1D positional embeddings (breaks particle permutation symmetry).")
    group.add_argument('--abs_time_emb', action='store_true', help="Enable absolute 1D time embeddings (critical for sequence ordering).")

    ## Transport arguments
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path_type", type=str, default="GVP", choices=["Linear", "GVP", "VP"], help="Type of flow matching path.")
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"], help="Target prediction type for the network.")
    group.add_argument("--sampling_method", type=str, default="dopri5", choices=["dopri5", "euler"], help="ODE solver method for sampling.")
    
    args = parser.parse_args()
    
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)
    
    # RETURN BOTH ARGS AND THE PARSER
    return args, parser