import random

from torch.utils.data import DataLoader


def sample_validation_starts(num_frames, chain_length, num_chains, gap, seed):
    max_start = num_frames - chain_length
    if max_start < 0:
        raise ValueError('Not enough frames for validation chain')
    rng = random.Random(seed)
    candidates = list(range(max_start + 1))
    rng.shuffle(candidates)
    selected = []
    for start in candidates:
        end = start + chain_length - 1
        if all(end < other_start - gap or start > other_end + gap for other_start, other_end in selected):
            selected.append((start, end))
            if len(selected) == num_chains:
                break
    if len(selected) != num_chains:
        raise ValueError('Failed to allocate required number of non-overlapping validation chains')
    selected.sort()
    return [start for start, _ in selected]


def build_train_starts(num_frames, sequence_length, val_starts, val_chain_length, gap):
    max_start = num_frames - sequence_length
    if max_start < 0:
        raise ValueError('Not enough frames for training chain')
    starts = []
    for start in range(max_start + 1):
        end = start + sequence_length - 1
        valid = True
        for val_start in val_starts:
            val_end = val_start + val_chain_length - 1
            if not (end < val_start - gap or start > val_end + gap):
                valid = False
                break
        if valid:
            starts.append(start)
    if not starts:
        raise ValueError('No training chains left after validation allocation')
    return starts


def make_sequential_splits(num_frames, sequence_length, val_chain_length, num_val_chains, gap, seed):
    val_starts = sample_validation_starts(num_frames, val_chain_length, num_val_chains, gap, seed)
    train_starts = build_train_starts(num_frames, sequence_length, val_starts, val_chain_length, gap)
    return train_starts, val_starts


def make_loader(dataset, batch_size, num_workers, shuffle, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
