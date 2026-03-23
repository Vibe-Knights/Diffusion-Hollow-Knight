import math


def scheduled_value(epoch, start_value, end_value, ramp_epochs, mode='linear'):
    if ramp_epochs <= 0:
        return end_value
    progress = min(1.0, max(0.0, epoch / ramp_epochs))
    if mode == 'cosine':
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start_value + (end_value - start_value) * progress
