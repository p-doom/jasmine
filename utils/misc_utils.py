from jax.tree_util import tree_map, tree_reduce
import optax



def get_lr_schedule(lr_schedule, init_lr, max_lr, final_lr, total_steps, warmup_steps, wsd_decay_steps):
    supported_schedules = ["wsd", "cos", "const"]
    assert lr_schedule in supported_schedules, f"Learning rate schedule not supported. Please use one of {supported_schedules}"
    if lr_schedule == "const":
        return optax.constant_schedule(max_lr)
    if lr_schedule == "cos":
        assert warmup_steps <= total_steps, "Warmup steps can't be greater than total steps."
        return optax.warmup_cosine_decay_schedule(
            init_value=init_lr,
            peak_value=max_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps, # Note: decay_steps includes the warmup steps, so we need to pass total value
            end_value=final_lr 
        )
    if lr_schedule == "wsd":
        assert warmup_steps + wsd_decay_steps <= total_steps, "Warmup and decay period is longer than total steps."
        schedules = [
            optax.linear_schedule(init_value=init_lr, end_value=max_lr, transition_steps=warmup_steps),
            optax.constant_schedule(value=max_lr),
            optax.linear_schedule(init_value=max_lr, end_value=final_lr, transition_steps=wsd_decay_steps),
        ]
        boundaries = [warmup_steps, total_steps - wsd_decay_steps]
        return optax.join_schedules(schedules, boundaries)


def _count_leaf(x):
    """Count parameters in a single leaf node."""
    if hasattr(x, "size"):
        return x.size
    return 0


def _count_component(component_params):
    """Count total parameters in a component."""
    return tree_reduce(
        lambda x, y: x + y, tree_map(_count_leaf, component_params), initializer=0
    )


def count_parameters_by_component(params):
    """Count parameters for each component of the model.

    Args:
        params: Model parameters dictionary
        component_names: List of component names to count. If None, counts all components.

    Returns:
        Dictionary with parameter counts for each component
    """

    component_names = list(params["params"].keys())
    print(f"Counting all components: {component_names}")

    counts = {}
    total_params = 0

    for name in component_names:
        if "params" in params and name in params["params"]:
            component_params = params["params"][name]
        else:
            component_params = params

        count = _count_component(component_params)
        counts[name] = count
        total_params += count

    counts["total"] = total_params
    return counts
