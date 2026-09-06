from .ema import ExponentialMovingAverage
from .logger import get_logger

logger = get_logger(__name__)

import pytorch_lightning as pl
import torch, time, os, wandb
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial

from .latent_model.latent_model import LatentMDGenModel
from .transport.transport import create_transport, Sampler
from .tensor_utils import tensor_tree_map, get_offsets, get_absolute
from .rigid_utils import Rigid, quat_multiply, rot_vec_mul, quat_to_rot


def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log

def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.nanmean(log[key])
        except:
            pass
    return out

class Wrapper(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = 0

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.mean().item()
        log = self._log
        if self.stage == 'train' or self.args.validate:
            log["iter_" + key].append(data)
        log[self.stage + "_" + key].append(data)

    def load_ema_weights(self):
        logger.info('Loading EMA weights')
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        logger.info('Restoring cached weights')
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def on_before_zero_grad(self, *args, **kwargs):
        if self.args.ema:
            self.ema.update(self.model)

    def training_step(self, batch, batch_idx):
        if self.args.ema and (self.ema.device != self.device):
            self.ema.to(self.device)
        return self.general_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        if self.args.ema:
            if (self.ema.device != self.device):
                self.ema.to(self.device)
            if (self.cached_weights is None):
                self.load_ema_weights()
        self.general_step(batch, stage='val')
        if self.args.validate and self.iter_step % self.args.print_freq == 0:
            self.print_log()

    def on_train_epoch_end(self):
        self.print_log(prefix='train', save=False)

    def on_validation_epoch_end(self):
        if self.args.ema:
            self.restore_cached_weights()
        self.print_log(prefix='val', save=False)

    def on_before_optimizer_step(self, optimizer):
        if (self.trainer.global_step + 1) % self.args.print_freq == 0:
            self.print_log()

    def on_load_checkpoint(self, checkpoint):
        if self.args.ema:
            self.ema.load_state_dict(checkpoint["ema"])

    def on_save_checkpoint(self, checkpoint):
        if self.args.ema:
            if self.cached_weights is not None:
                self.restore_cached_weights()
            checkpoint["ema"] = self.ema.state_dict()

    def print_log(self, prefix='iter', save=False, extra_logs=None):
        log = self._log
        log = {key: log[key] for key in log if f"{prefix}_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)

        mean_log.update({
            'epoch': self.trainer.current_epoch,
            'trainer_step': self.trainer.global_step + int(prefix == 'iter'),
            'iter_step': self.iter_step,
        })
        
        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.args.wandb:
                wandb.log(mean_log)
        
        for key in list(log.keys()):
            if f"{prefix}_" in key:
                del self._log[key]

    def configure_optimizers(self):
        cls = torch.optim.AdamW if self.args.adamW else torch.optim.Adam
        return cls(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)


class NewMDGenWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)
        
        self.latent_dim = 7 # 3 translation + 4 quaternion
        self.model = LatentMDGenModel(args, self.latent_dim)

        self.transport = create_transport(
            args,
            args.path_type,
            args.prediction,
            None, 
        )
        self.transport_sampler = Sampler(self.transport)

        if not hasattr(args, 'ema'):
            args.ema = False
        if args.ema: 
            self.ema = ExponentialMovingAverage(model=self.model, decay=args.ema_decay)
            self.cached_weights = None

    def prep_batch(self, batch):
        """
        Prepares the batch for forward simulation using relative offsets.
        Uses pure tensor operations to bypass slow Eigenvalue decompositions.
        """
        
        trans = batch['trans'].float() # absolute translations in global frame, shape (B, T, N, 3)
        rots = batch['rots'].float() # absolute quaternions in global frame, shape (B, T, N, 4)
        mask = batch['mask']
        
        B, T, N, _ = trans.shape
        
        # 1. Concatenate to form the 7D latents
        trajectory = torch.cat([rots, trans], dim=-1)

        # 2. Get absolute Rigids (Required for the IPA layers' start_frames)
        rigids = Rigid.from_tensor_7(trajectory, normalize_quats=True)

        # 3. PURE TENSOR OFFSET CALCULATION (Bypasses linalg.eigh bottleneck)
        offsets = get_offsets(rots, trans)
        
        # 4. Create Loss and Conditioning Masks. 
        loss_mask = mask.unsqueeze(-1).expand(-1, -1, -1, 7) # mask and loss_mask are always 1

        # Forward Simulation Condition: Only the very first frame (T=0) is fully known
        cond_mask = torch.zeros(B, T, N, dtype=torch.int, device=trajectory.device)
        cond_mask[:, 0] = 1 

        return {
            'latents': offsets, 
            'loss_mask': loss_mask,
            'model_kwargs': {
                'start_frames': rigids[:, 0], # Provide the initial absolute frame geometry to IPA
                'mask': mask,
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), offsets, 0.0),
                'x_cond_mask': cond_mask,
            }
        }
    

    def general_step(self, batch, stage='train'):
        self.iter_step += 1
        self.stage = stage
        start = time.time()

        prep = self.prep_batch(batch)

        # Compute flow matching loss
        out_dict = self.transport.training_losses(
            model=self.model,
            x1=prep['latents'],
            mask=prep['loss_mask'],
            model_kwargs=prep['model_kwargs']
        )
        
        loss = out_dict['loss']
        self.log('loss', loss)
        self.log('time', out_dict['t'])
        self.log('general_step_dur', time.time() - start)
        self.last_log_time = time.time()
        
        return loss.mean()

    def inference(self, batch):
        """
        Runs the reverse ODE solver to generate a trajectory, 
        then converts the relative offsets back to absolute coordinates
        using pure tensor math to avoid linalg.eigh bottlenecks.
        """
                
        prep = self.prep_batch(batch)
        
        latents = prep['latents'] # offsets in relative space, shape (B, T, N, 7)
        B, T, N, _ = latents.shape

        # 1. Sample from prior distribution N(0, I)
        zs = torch.randn(B, T, N, self.latent_dim, device=self.device)

        # 2. Get the sampling function (e.g., dopri5 or euler ODE solver)
        sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method)
        
        # 3. Solve the ODE path
        samples = sample_fn(
            zs,
            self.model,
            **prep['model_kwargs']
        )[-1] # offsets in relative space, shape (B, T, N, 7)
        
        # 'samples' now contains the generated *relative* offsets of shape (B, T, N, 7)
        q_rel = samples[..., :4]
        t_rel = samples[..., 4:7]
        
        # 5. Retrieve the absolute reference frames (T=0)
        ref_rigids = prep['model_kwargs']['start_frames']

        absolute_trajectory = get_absolute(q_rel, t_rel, ref_rigids)
        
        return absolute_trajectory