from dataclasses import dataclass
import os
import time

import einops
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import flax.jax_utils
import optax
import orbax
from orbax.checkpoint import PyTreeCheckpointer
import numpy as np
import dm_pix as pix
import jax
import jax.numpy as jnp
import tyro
import wandb

from models.tokenizer import TokenizerVQVAE
from utils.dataloader import get_dataloader

ts = int(time.time())


@dataclass
class Args:
    # Experiment
    num_steps: int = 300_000
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_resolution: int = 64
    data_dir: str = "data/coinrun_episodes"
    checkpoint: str = ""
    # Optimization
    vq_beta: float = 0.25
    batch_size: int = 48  # This will be the global batch size
    min_lr: float = 3e-4
    max_lr: float = 3e-4
    warmup_steps: int = 10000
    # Tokenizer
    model_dim: int = 512
    latent_dim: int = 32
    num_latents: int = 1024
    patch_size: int = 4
    num_blocks: int = 8
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.01
    # Logging
    log: bool = False
    entity: str = ""
    project: str = ""
    log_interval: int = 5
    log_image_interval: int = 250
    ckpt_dir: str = ""
    log_checkpoint_interval: int = 10000
    log_gradients: bool = False


args = tyro.cli(Args)


def tokenizer_loss_fn(params, state, inputs):
    outputs = state.apply_fn(
        params, inputs, training=True, rngs={"dropout": inputs["rng"]}
    )
    mse = jnp.square(inputs["videos"] - outputs["recon"]).mean()
    q_loss = jnp.square(jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]).mean()
    commitment_loss = jnp.square(
        outputs["emb"] - jax.lax.stop_gradient(outputs["z"])
    ).mean()
    loss = mse + q_loss + args.vq_beta * commitment_loss

    gt = inputs["videos"].clip(0, 1).reshape(-1, *inputs["videos"].shape[2:])
    recon = outputs["recon"].clip(0, 1).reshape(-1, *outputs["recon"].shape[2:])
    psnr = pix.psnr(gt, recon).mean()
    ssim = pix.ssim(gt, recon).mean()
    _, index_counts = jnp.unique_counts(
        jnp.ravel(outputs["indices"]), size=args.num_latents, fill_value=0
    )
    codebook_usage = (index_counts != 0).mean()
    metrics = dict(
        loss=loss,
        mse=mse,
        q_loss=q_loss,
        commitment_loss=commitment_loss,
        psnr=psnr,
        ssim=ssim,
        codebook_usage=codebook_usage,
    )
    return loss, (outputs["recon"], metrics)


def train_step_for_pmap(state, inputs):
    grad_fn = jax.value_and_grad(tokenizer_loss_fn, has_aux=True, allow_int=True)
    (loss, (recon, metrics)), grads = grad_fn(state.params, state, inputs)

    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    metrics = jax.lax.pmean(metrics, axis_name='batch')

    state = state.apply_gradients(grads=grads)

    if args.log_gradients:
        if 'params' in grads:
            if 'encoder' in grads['params']:
                 metrics["encoder_gradients_std/"] = jax.tree_map(
                    lambda x: x.std(), grads["params"]["encoder"]
                )
            if 'vq' in grads['params']:
                metrics["vq_gradients_std/"] = jax.tree_map(
                    lambda x: x.std(), grads["params"]["vq"]
                )
            if 'decoder' in grads['params']:
                metrics["decoder_gradients_std/"] = jax.tree_map(
                    lambda x: x.std(), grads["params"]["decoder"]
                )
        else:
            print("Warning: 'params' key not found in gradients, cannot log gradient std.")
            
    return state, loss, recon, metrics


if __name__ == "__main__":
    print("Pinpoint print: Starting tokenizer training script with pmap...") # Keep this for pinpointing
    print(f"Pinpoint print: JAX version: {jax.__version__}")
    print(f"Pinpoint print: JAX backend: {jax.default_backend()}")
    try:
        # Suppress the DeprecationWarning for this specific call if possible, or acknowledge it.
        # For now, we'll let it print if it occurs.
        backend_instance = jax.extend.backend.get_backend() if hasattr(jax.extend, 'backend') else jax.lib.xla_bridge.get_backend()
        print(f"Pinpoint print: JAX platform version (CUDA/ROCm): {backend_instance.platform_version}")
    except Exception as e:
        print(f"Pinpoint print: Could not get JAX platform version: {e}")

    num_devices = jax.local_device_count()
    print(f"Pinpoint print: Number of local JAX devices found: {num_devices}")
    if num_devices == 0:
        print("Error: No JAX devices found. Exiting. Ensure JAX is installed correctly for your hardware (GPU/TPU).")
        exit(1)
    
    if num_devices > 0: # Changed from > 0 to ensure it runs even for single device for testing, though pmap over 1 device is trivial
        try:
            print(f"Pinpoint print: Attempting a simple pmap operation on {num_devices} devices to pre-initialize communication.")
            
            # Define the function to be pmapped
            def _simple_comms_test_fn(x):
                # Perform a simple computation involving an axis index and a collective (pmean)
                # The axis_name here must match the one passed to jax.pmap
                return jax.lax.pmean(x + jax.lax.axis_index('test_axis'), axis_name='test_axis')

            # Apply jax.pmap, passing the function and the axis_name
            simple_comms_test_pmapped = jax.pmap(_simple_comms_test_fn, axis_name='test_axis')
            
            dummy_host_data = np.arange(num_devices, dtype=np.float32).reshape(num_devices, 1)
            result = simple_comms_test_pmapped(dummy_host_data)
            print(f"Pinpoint print: Simple pmap communication test successful. Result (all devices should show same mean): {result[0]}")
        except Exception as e:
            print(f"CRITICAL: Error during simple pmap communication test: {e}")
            print("This suggests a fundamental issue with JAX pmap/NCCL setup on your system (e.g., driver, CUDA, NCCL versions).")
            print("Please check your environment and JAX installation.")
            print("Try running with NCCL_DEBUG=INFO for more NCCL logs.")
            import traceback
            traceback.print_exc() # Print full traceback for the pmap test error
            exit(1) 

    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size ({args.batch_size}) must be divisible by "
            f"the number of devices ({num_devices})."
        )
    per_device_batch_size = args.batch_size // num_devices
    print(f"Using {num_devices} devices with per-device batch size of {per_device_batch_size}.")

    rng = jax.random.PRNGKey(args.seed)
    if args.log:
        wandb.init(entity=args.entity, project=args.project, group="debug", config=args)

    tokenizer = TokenizerVQVAE(
        in_dim=args.image_channels,
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        patch_size=args.patch_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        codebook_dropout=args.codebook_dropout,
    )
    rng, init_rng = jax.random.split(rng)
    image_shape = (args.image_resolution, args.image_resolution, args.image_channels)
    dummy_init_videos = jnp.zeros(
        (args.batch_size, args.seq_len, *image_shape), dtype=jnp.float32 
    )
    init_inputs = dict(videos=dummy_init_videos)
    init_params_tree = tokenizer.init(init_rng, init_inputs)

    step = 0
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        try:
            restored_contents = PyTreeCheckpointer().restore(args.checkpoint)
            if "model" in restored_contents and isinstance(restored_contents["model"], dict) and "params" in restored_contents["model"]:
                init_params_tree = restored_contents["model"]["params"] 
                print("Successfully loaded model parameters (init_params_tree) from checkpoint.")
            elif "model" in restored_contents and hasattr(restored_contents["model"], "params"): # e.g. if model is a TrainState
                init_params_tree = restored_contents["model"].params
                print("Successfully loaded model parameters (init_params_tree) from checkpoint (TrainState object).")
            else:
                print("Warning: Checkpoint structure not as expected. 'model' or 'model[params]' key missing. Using fresh parameters.")
            
            parsed_step = int(args.checkpoint.split("_")[-1])
            step = parsed_step 
            print(f"Resuming from step {step} based on checkpoint filename.")
        except ValueError:
            print("Could not parse step from checkpoint filename. Checkpoint parameters loaded, but step count might be inaccurate if not stored in checkpoint itself.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Using fresh parameters and starting from step 0.")
            init_params_tree = tokenizer.init(init_rng, init_inputs) 
            step = 0

    lr_schedule = optax.warmup_cosine_decay_schedule(
        args.min_lr, args.max_lr, args.warmup_steps, args.num_steps
    )
    tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.9, weight_decay=1e-4)
    train_state = TrainState.create(apply_fn=tokenizer.apply, params=init_params_tree, tx=tx)

    train_state = flax.jax_utils.replicate(train_state, devices=jax.local_devices())
    print("Train state replicated across devices.")

    # This is the correct way to call pmap when you need to specify axis_name
    p_train_step = jax.pmap(train_step_for_pmap, axis_name='batch')
    print("p_train_step created.")

    dataloader = get_dataloader(args.data_dir, args.seq_len, args.batch_size) 
    print("Dataloader initialized.")

    while step < args.num_steps:
        for videos_batch in dataloader: 
            if videos_batch.shape[0] != args.batch_size:
                print(f"Skipping incomplete batch of size {videos_batch.shape[0]} (expected {args.batch_size}).")
                continue 

            sharded_videos = einops.rearrange(videos_batch, '(d b) ... -> d b ...', d=num_devices)
            
            rng, step_rng_master = jax.random.split(rng)
            sharded_dropout_keys = jax.random.split(step_rng_master, num_devices)
            
            sharded_inputs = dict(videos=sharded_videos, rng=sharded_dropout_keys)

            train_state, loss, recon, metrics = p_train_step(train_state, sharded_inputs)
            
            current_loss = loss[0] 
            current_metrics = jax.tree_map(lambda x: x[0], metrics)

            if step % args.log_interval == 0: 
                 print(f"Step {step}/{args.num_steps}, Global Loss: {current_loss:.4f}")
            step += 1

            if args.log:
                if step % args.log_interval == 0:
                    log_data = {"loss": current_loss, "step": step}
                    for k, v in current_metrics.items():
                        # Ensure metrics are scalar for wandb logging
                        if hasattr(v, 'item'): # Check if it has an item() method (like JAX/NumPy arrays)
                            log_data[k] = v.item() 
                        else:
                            log_data[k] = v # Assume it's already a Python scalar
                    wandb.log(log_data)
                
                if step % args.log_image_interval == 0:
                    gt_seq_device0_sample0 = sharded_inputs["videos"][0][0] 
                    recon_seq_device0_sample0 = recon[0][0].clip(0, 1)    
                    
                    comparison_seq = jnp.concatenate((gt_seq_device0_sample0, recon_seq_device0_sample0), axis=1) 
                    comparison_seq = einops.rearrange(
                        comparison_seq * 255, "t h w c -> h (t w) c"
                    ) 
                    
                    log_images_payload = dict(
                        image=wandb.Image(np.asarray(gt_seq_device0_sample0[0])), 
                        recon=wandb.Image(np.asarray(recon_seq_device0_sample0[0])), 
                        true_vs_recon_seq=wandb.Image(
                            np.asarray(comparison_seq.astype(np.uint8))
                        ),
                    )
                    wandb.log(log_images_payload)
                
                if step % args.log_checkpoint_interval == 0:
                    unreplicated_train_state = flax.jax_utils.unreplicate(train_state)
                    ckpt = {"model": unreplicated_train_state} 
                    
                    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                    save_args_orbax = orbax_utils.save_args_from_target(ckpt)
                    
                    # Construct checkpoint path
                    if args.ckpt_dir: # Only create if ckpt_dir is specified
                        os.makedirs(args.ckpt_dir, exist_ok=True)
                        checkpoint_path = os.path.join(args.ckpt_dir, f"tokenizer_{ts}_{step}")
                    else: # Save in current directory if ckpt_dir is not specified
                        checkpoint_path = f"tokenizer_{ts}_{step}"
                    
                    orbax_checkpointer.save(
                        checkpoint_path,
                        ckpt,
                        save_args=save_args_orbax,
                        force=True 
                    )
                    print(f"Saved checkpoint to {checkpoint_path} at step {step}")

            if step >= args.num_steps:
                break
    
    print("Training finished.")
    if args.log:
        wandb.finish()
