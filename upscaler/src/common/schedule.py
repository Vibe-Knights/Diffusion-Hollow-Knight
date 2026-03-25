import math


def scheduled_value(epoch, start_value, end_value, ramp_epochs, mode='linear'):
    if ramp_epochs <= 0:
        return end_value
    progress = min(1.0, max(0.0, epoch / ramp_epochs))
    if mode == 'cosine':
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start_value + (end_value - start_value) * progress


class Schedule:
    def __init__(self, start, end, ramp_epochs, mode='linear', delay=0):
        self.start = float(start)
        self.end = float(end)
        self.ramp_epochs = int(ramp_epochs)
        self.mode = str(mode)
        self.delay = int(delay)

    def get(self, epoch: int) -> float:
        effective = max(0, epoch - self.delay)
        return scheduled_value(effective, self.start, self.end, self.ramp_epochs, self.mode)
