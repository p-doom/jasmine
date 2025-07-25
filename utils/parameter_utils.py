from jax.tree_util import tree_map, tree_reduce


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
