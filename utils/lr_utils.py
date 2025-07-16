import optax

def get_lr_schedule(lr_schedule, init_lr, max_lr, decay_end, total_steps, warmup_steps, wsd_decay_steps):
    supported_schedules = ["wsd", "cos"]
    assert lr_schedule in supported_schedules, f"Learning rate schedule not supported. Please use one of {supported_schedules}"
    if lr_schedule == "cos":
        assert warmup_steps <= total_steps, "Warmup steps can't be greater than total steps."
        return optax.warmup_cosine_decay_schedule(
            init_value=init_lr,
            peak_value=max_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps, # Note: decay_steps includes the warmup steps, so we need to pass total value
            end_value=decay_end 
        )
    if lr_schedule == "wsd":
        assert warmup_steps + wsd_decay_steps <= total_steps, "Warmup and decay period is longer than total steps."
        schedules = [
            optax.linear_schedule(init_value=init_lr, end_value=max_lr, transition_steps=warmup_steps),
            optax.constant_schedule(value=max_lr),
            optax.linear_schedule(init_value=max_lr, end_value=decay_end, transition_steps=wsd_decay_steps),
        ]
        boundaries = [warmup_steps, total_steps - wsd_decay_steps]
        return optax.join_schedules(schedules, boundaries)
