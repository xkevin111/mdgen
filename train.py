import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric")
import torch, os, wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

# Use local imports to avoid relative import errors
from model.parsing import parse_train_args
from model.logger import get_logger, log_args_and_help
from model.dataset import MDTrajectoryDataset
from model.wrapper import NewMDGenWrapper

logger = get_logger(__name__)

if __name__ == '__main__':
    # Now unpacking BOTH args and parser
    args, parser = parse_train_args() 

    log_args_and_help(logger, args, parser)

    torch.set_float32_matmul_precision('medium') 

    if args.wandb: 
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "default_entity"),
            settings=wandb.Settings(start_method="fork"),
            project="ellipsoid_mdgen",
            name=args.run_name,
            config=args, 
        )

    # Define batches safely
    train_b = args.train_batches if args.train_batches is not None else 1000
    val_b = args.val_batches if args.val_batches is not None else 100

    trainset = MDTrajectoryDataset(args=args, repeat=train_b * args.batch_size)

    if args.overfit:
        valset = trainset    
    else:
        valset = MDTrajectoryDataset(args=args, repeat=val_b * args.batch_size)

    train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    pin_memory=True,            # Speeds up CPU to GPU transfers
    persistent_workers=True,    # Keeps workers alive between epochs
    prefetch_factor=2           # Prefetches batches
    )

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,            # Speeds up CPU to GPU transfers
        persistent_workers=True,    # Keeps workers alive between epochs
        prefetch_factor=2           # Prefetches batches
    )

    model = NewMDGenWrapper(args)
        
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else 'auto',
        profiler="simple", 
        max_epochs=args.epochs,
        limit_train_batches=args.train_batches or 1.0,
        limit_val_batches=0.0 if args.no_validate else (args.val_batches or 1.0),
        num_sanity_val_steps=0,
        precision=args.precision,
        enable_progress_bar=not args.wandb,
        gradient_clip_val=args.grad_clip,
        default_root_dir=os.environ.get("MODEL_DIR", "./saved_models"), 
        callbacks=[
            ModelCheckpoint(
                dirpath=os.environ.get("MODEL_DIR", "./saved_models"), 
                save_top_k=-1,
                every_n_epochs=args.ckpt_freq,
            ),
            ModelSummary(max_depth=2),
        ],
        accumulate_grad_batches=args.accumulate_grad,
        val_check_interval=args.val_freq,
        check_val_every_n_epoch=args.val_epoch_freq,
        logger=False
    )

    if args.validate:
        trainer.validate(model, val_loader, ckpt_path=args.ckpt)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)

# python train.py --run_name ellipsoid_symmetric --batch_size 2 --accumulate_grad 8 --num_frames 50 --use_rope_t --abs_time_emb --num_workers 4 --box 11.7528 12.438761725573098 11.849512010564649 --overfit_frame
# python train.py --run_name ellipsoid_symmetric --batch_size 2 --accumulate_grad 8 --num_frames 50 --use_rope_t --use_rope_l --abs_pos_emb --abs_time_emb --num_workers 4 --overfit_frame --crop 60 --train_batches 100 --val_batches 10 --prepend_ipa