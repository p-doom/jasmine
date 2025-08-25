import jax
import optax
from jax.tree_util import tree_map, tree_reduce


def get_lr_schedule(
    lr_schedule: str,
    init_lr: float,
    max_lr: float,
    decay_end: float,
    total_steps: int,
    warmup_steps: int,
    wsd_decay_steps: int,
) -> optax.Schedule:
    supported_schedules = ["wsd", "cos"]
    if lr_schedule == "cos":
        assert (
            warmup_steps <= total_steps
        ), "Warmup steps can't be greater than total steps."
        return optax.warmup_cosine_decay_schedule(
            init_value=init_lr,
            peak_value=max_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,  # Note: decay_steps includes the warmup steps, so we need to pass total value
            end_value=decay_end,
        )
    elif lr_schedule == "wsd":
        assert (
            warmup_steps + wsd_decay_steps <= total_steps
        ), "Warmup and decay period is longer than total steps."
        schedules = [
            optax.linear_schedule(
                init_value=init_lr, end_value=max_lr, transition_steps=warmup_steps
            ),
            optax.constant_schedule(value=max_lr),
            optax.linear_schedule(
                init_value=max_lr, end_value=decay_end, transition_steps=wsd_decay_steps
            ),
        ]
        boundaries = [warmup_steps, total_steps - wsd_decay_steps]
        return optax.join_schedules(schedules, boundaries)
    else:
        raise ValueError(
            f"Learning rate schedule not supported. Please use one of {supported_schedules}"
        )


def _count_component(component_params):
    """Count total parameters in a component."""
    params_sizes = jax.tree.map(jax.numpy.size, component_params)
    total_parameters = jax.tree.reduce(lambda x, y: x + y, params_sizes)
    return total_parameters


def count_parameters_by_component(params):
    """Count parameters for each component of the model.

    Args:
        params: Model parameters from nnx.split(model, nnx.Param, ...)

    Returns:
        Dictionary with parameter counts for each component
    """
    component_names = list(params.keys())
    print(f"Counting all components: {component_names}")

    counts = {}
    total_params = 0

    for name in component_names:
        component_params = params[name]
        count = _count_component(component_params)
        counts[name] = count
        total_params += count

    counts["total"] = total_params
    return counts


def bytes_to_gb(num_bytes):
    return num_bytes / (1024**3)


def print_compiled_memory_stats(compiled_stats):
    """from: https://github.com/AI-Hypercomputer/maxtext/blob/b18829fbaa48aec7ac350a03e62248e24c6a76b2/MaxText/max_utils.py#L739"""
    output_gb = bytes_to_gb(compiled_stats.output_size_in_bytes)
    temp_gb = bytes_to_gb(compiled_stats.temp_size_in_bytes)
    argument_gb = bytes_to_gb(compiled_stats.argument_size_in_bytes)
    alias_gb = bytes_to_gb(compiled_stats.alias_size_in_bytes)
    host_temp_gb = bytes_to_gb(compiled_stats.host_temp_size_in_bytes)
    total_gb = output_gb + temp_gb + argument_gb - alias_gb
    print(
        f"Total memory size: {total_gb:.1f} GB, Output size: {output_gb:.1f} GB, Temp size: {temp_gb:.1f} GB, "
        f"Argument size: {argument_gb:.1f} GB, Host temp size: {host_temp_gb:.1f} GB."
    )


def print_compiled_cost_analysis(cost_stats):
    flops = float(cost_stats.get("flops", 0.0))
    bytes_accessed = float(cost_stats.get("bytes accessed", 0.0))
    gb = bytes_to_gb(bytes_accessed) if bytes_accessed else 0.0
    intensity = (flops / bytes_accessed) if bytes_accessed else float("nan")
    print(
        f"FLOPs: {flops:.3e}, Bytes: {bytes_accessed:.3e} ({gb:.1f} GB), "
        f"Intensity: {intensity:.1f} FLOPs/byte"
    )


def print_mem_stats(label: str):
    """from: https://github.com/AI-Hypercomputer/maxtext/blob/7898576359bacde81be25cb3038e348aac1f943b/MaxText/max_utils.py#L713"""
    print(f"\nMemstats: {label}:")
    try:
        for d in jax.local_devices():
            stats = d.memory_stats()
            used = round(stats["bytes_in_use"] / 2**30, 2)
            limit = round(stats["bytes_limit"] / 2**30, 2)
            print(f"\tUsing (GB) {used} / {limit} ({used/limit:%}) on {d}")
    except (RuntimeError, KeyError, TypeError) as ex:
        print(f"\tMemstats unavailable, error: {ex}")
