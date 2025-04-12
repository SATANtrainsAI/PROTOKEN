import os
import math  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import GradScaler, autocast
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision.utils import make_grid, save_image
from pathlib import Path
from einops import rearrange
from tqdm import tqdm
import argparse
from contextlib import nullcontext
import time


# Import custom utilities and models
import wandb
from dataloader import train_for_bin, train_for_all
from model import VQVAE



def exists(val):
    return val is not None

def print0(*args, **kwargs):
    """
    Modified print that only prints from the master process.
    If not a distributed run, it behaves like a regular print.
    """
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)




# ----------------------------
# Argument Parsing
# ----------------------------

parser = argparse.ArgumentParser(description="Train VQGAN VAE with DDP and Progressive GAN Integration")

# Initialization Arguments
parser.add_argument('--init_from', type=str, choices=['scratch', 'resume'], default='resume', help='Initialize from scratch or resume.')
parser.add_argument('--backend', type=str, default='nccl', help='Backend for distributed training.')
parser.add_argument('--dtype', type=str, choices=['float32', 'bfloat16', 'float16'], default='bfloat16', help='Data type for model weights.')


# Training Arguments
parser.add_argument('--max_iter', type=int, default=10000000, help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
parser.add_argument('--folder', type=str, default="/scratch/xwang213/img", help='Path to training images folder')
parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--valid_frac', type=float, default=0.0001, help='Validation fraction')
parser.add_argument('--out_dir', type=str, default="res_final", help='Folder to save results')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay factor.')
parser.add_argument('--lr', type=float, default=9e-4, help='Learning rate.')
parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer.')
parser.add_argument('--beta2', type=float, default=0.95, help='Beta2 for Adam optimizer.')
parser.add_argument('--warmup_iters', type=int, default=100, help='Number of warmup iterations.')
parser.add_argument('--lr_decay_iters', type=int, default=20000, help='Number of iterations for learning rate decay.')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value.')
parser.add_argument('--random_split_seed', type=int, default=42, help='Seed for data split')
parser.add_argument('--amp', type=bool, default=True, help='Use Automatic Mixed Precision')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training')
parser.add_argument('--log_interval', type=int, default=1, help='Iterations between logging.')
parser.add_argument('--eval_interval', type=int, default=500, help='Iterations between evaluations.')
parser.add_argument('--always_save_checkpoint', type=bool, default=True, help='Always save checkpoint.')
parser.add_argument('--eval_iters', type=int, default=10, help='Number of evaluation iterations.')
parser.add_argument('--num_epoch_per_bin', type=int, default=1, help='Number of epochs per bin.')




# DDP Argument
parser.add_argument('--ddp', type=bool, default=True, help='Use Distributed Data Parallel (DDP).')

args = parser.parse_args()



if args.ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    master = ddp_rank == 0
    seed_offset = ddp_rank
    print0(f"Initialized DDP: rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}")
else:
    master = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print0(f"Single-process setup. Rank: {ddp_rank}/{ddp_world_size}, Device: {device}")


if master:
    os.makedirs(args.out_dir, exist_ok=True)
    print0(f"Results will be saved at {args.out_dir}!")


if master:
    wandb.init(project="vqgan_vae_4096_res",
               name="vqvae_run",
               config=vars(args)) 

# ----------------------------
# Set Seeds and CUDA Configurations
# ----------------------------

torch.manual_seed(2001 + seed_offset)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2001 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True      # Allow TF32 on cuDNN

device_type = "cuda" if "cuda" in device else "cpu"


ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == "cpu" else autocast(enabled=args.amp, dtype=ptdtype, device_type="cuda")

from dataclasses import dataclass
@dataclass
class ModelArgs:
    """
    Hyperparameters for the model.
    Adjust to your preference. Typically n_embd ~ 256..768,
    n_head ~ 4..16, and so on, depending on memory and resolution.
    """
    # Basic
    n_embd: int = 1024
    mlp_hidden_dim: int = int(1024 * 2)
    n_head: int = 16
    latent_dim: int = 256   # dimension for Q/K/V
    rope_dim: int = 8
    is_causal: bool = False  # Usually no causal mask for VQ-VAE

    # Codebook
    q_channels: int = 1024
    codebook_dim: int = 1024
    codebook_size: int = 4096

    # Depth
    n_layer_encoder: int = 2
    n_layer_decoder: int = 2
    

    beta: float = 0.25


scaling_rates =  [2, 2, 2, 2, 2, 2]

model_args = dict(
    n_layer_encoder=ModelArgs.n_layer_encoder,
    n_layer_decoder=ModelArgs.n_layer_decoder,
    n_head=ModelArgs.n_head,
    n_embd=ModelArgs.n_embd,
    is_causal = ModelArgs.is_causal,
    rope_dim = ModelArgs.rope_dim,
    mlp_hidden_dim = ModelArgs.mlp_hidden_dim,
    latent_dim = ModelArgs.latent_dim,
    q_channels=ModelArgs.q_channels,
    codebook_dim=ModelArgs.codebook_dim,
    codebook_size = ModelArgs.codebook_size,
    beta = ModelArgs.beta
)

best_val_loss = 1e9
print0(model_args)

if args.init_from == "scratch":
    print("Trainig new model form scratch!")
    config = ModelArgs(**model_args)
    model = VQVAE(config, scaling_rates=scaling_rates)
    model.to(device)
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), device_type = device_type)
    iter_num = 0

if args.init_from == "resume":
    print(f"Resuming training from {args.out_dir}")
    ckpt_path = os.path.join(args.out_dir, "ckpt_vqvae35000.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelArgs(**model_args)
    model = VQVAE(config, scaling_rates)
    
    # Load state dict with strict=False to allow missing keys (new modalities)
    state_dict = checkpoint["model"]
    
    # Initialize missing layers (new modalities) if necessary
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys after loading state dict: {missing_keys}")
        print("These are likely the new modality layers and will be randomly initialized.")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
        # These are the new layers; they are already initialized in the model's __init__
    model.to(device)
    # Load optimizer state dict
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), device_type = device_type)
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

print0("World size:", ddp_world_size)
scaler = torch.GradScaler(enabled = args.amp)

checkpoint = None

if args.ddp:
    model = DDP(model, device_ids = [ddp_local_rank], find_unused_parameters=False)



def get_lr(it, warmup_iters = 1000, lr = 6e-4, min_lr = 1e-6):
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    
    if it > args.lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (args.lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def cost(X, reconstruction, code_diffs, beta):
        r_loss, l_loss = reconstruction.sub(X).pow(2).mean(), sum(code_diffs)
        Loss =  r_loss + beta * l_loss
        return Loss, r_loss, l_loss


def estimate_loss(val_dl):
   
    model.eval()
    losses = []
    for idx, (X) in enumerate(val_dl):
        if idx >= args.eval_iters:
            break
        X = X.to(device)
        with ctx:
            reconstruction, code_diffs, _, _, _ = model(X)
            loss,_, _ = cost(X, reconstruction, code_diffs, config.beta)
            losses.append(loss.item())
    loss = sum(losses) / len(losses)
    model.train()
    return loss



def save_reconstructions(iter_num, model, device, out_dir, dl):
    """
    Save image reconstructions in the desired format:
    Each row contains the real image and its corresponding reconstruction side by side.
    """
    print0(f"Saving reconstructions at iteration {iter_num}...")
    model.eval()
    with torch.no_grad():
        try:
            imgs = next(dl)
        except StopIteration:
            imgs = next(dl)  # Cycle ensures this doesn't happen
        with ctx:

            imgs = imgs.to(device, non_blocking=True)
            recons, _, _, _, _ = model(imgs)

        # Interleave real and reconstructed images
        interleaved = torch.stack([imgs, recons], dim=1).reshape(-1, 3, imgs.shape[2], imgs.shape[3])

        # Create grid with 2 images per row
        grid = make_grid(interleaved, nrow=2, normalize=True, value_range=(-1, 1))
        save_image(grid, Path(out_dir) / f'recon_iter_{iter_num}.png')
    model.train()
    print0(f"Reconstructions saved at iteration {iter_num}.")




print0("Initializing Datasets and DataLoaders...")

root_folder = Path(args.folder)


t0 = time.time()
raw_model = model.module if args.ddp else model

    # Start from smallest bin => bin_40_64 => do 4000 steps => next bin => ...

dl, val_dl = train_for_all(
            root_folder,
            patch_size = scaling_rates[0],
            batch_size = args.batch_size,
            ddp = args.ddp,
            ddp_world_size = ddp_world_size,
            ddp_rank = ddp_rank,
            num_workers = 4,
            valid_frac = 0.01
        )
    

print0("start training")
while True:

        micro_step = 0

        optimizer.zero_grad(set_to_none = True)
        
        for X in dl:
            X = X.to(device)
          
            with ctx:
                reconstruction, code_diffs, _, _, _ = model(X)
                loss, r_loss, l_loss = cost(X, reconstruction, code_diffs, config.beta)
                loss = loss / args.grad_accum_steps
    

            scaler.scale(loss).backward()

            micro_step += 1

            if micro_step % args.grad_accum_steps == 0:
                lr = get_lr(iter_num)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                if args.ddp:
                    model.require_backward_grad_sync = True

                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none = True)

                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                if iter_num % args.log_interval == 0 and master:
                    lossf = loss.item() * args.grad_accum_steps
                    print(f"iter {iter_num}: loss {lossf:.4f}, r_loss {r_loss.item():.4f}, time {dt * 1000:.2f}ms")
                    wandb.log({
                    "train_loss": lossf,
                    "reconstruction_loss": r_loss.item(),
                    "learning_rate": lr,
                    "time_ms": dt * 1000
                }, step=iter_num)

                iter_num += 1

                if (iter_num % args.eval_interval == 0):
                    val_loss = estimate_loss(val_dl)
                    print(f"step {iter_num}: val loss {val_loss:.4f}")
                    if master:
                        wandb.log({"val_loss": val_loss}, step=iter_num)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        
                    print(f"saving checkpoint to {args.out_dir}")
                    checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                        }
                    torch.save(checkpoint, os.path.join(args.out_dir, f'ckpt_vqvae{iter_num}.pt'))
                    save_reconstructions(iter_num, model, device, args.out_dir, val_dl)
                        

if args.ddp:
    destroy_process_group()
