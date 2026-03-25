import torch
from torch.amp import autocast, GradScaler

from src.common.schedule import Schedule
from src.losses.manager import WeightedLossManager


class UpscalerTrainer:
    def __init__(self, model, optimizer, lr_scheduler, train_loader, device, max_epochs,
                 loss_manager: WeightedLossManager,
                 teacher_forcing: Schedule = None,
                 callbacks=None, use_amp=True, grad_clip=1.0,
                 rollout_length=None, flow_estimator=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.device = device
        self.max_epochs = max_epochs
        self.loss_manager = loss_manager.to(device)
        self.teacher_forcing = teacher_forcing
        self.callbacks = callbacks or []
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.grad_clip = grad_clip
        self.rollout_length = rollout_length
        self.flow_estimator = flow_estimator

        self.current_epoch = -1
        self.current_batch = -1
        self.batch_logs = {}
        self.train_logs = {}
        self.val_logs = {}
        self.epoch_logs = {}

    @torch.no_grad()
    def _compute_flow(self, lr_prev, lr_curr):
        if self.flow_estimator is None:
            return None
        return self.flow_estimator.calc_batch(lr_prev, lr_curr)

    def _train_epoch(self):
        self.model.train()
        loss_sum = 0.0
        teacher_p = self.teacher_forcing.get(self.current_epoch) if self.teacher_forcing else 0.0

        for batch_idx, (lr_seq, target_seq) in enumerate(self.train_loader):
            lr_seq = lr_seq.to(self.device, non_blocking=True)
            target_seq = target_seq.to(self.device, non_blocking=True)
            steps = min(lr_seq.shape[1], self.rollout_length or lr_seq.shape[1])

            self.optimizer.zero_grad(set_to_none=True)
            prev_pred = None
            total_loss = torch.tensor(0.0, device=self.device)

            for t in range(steps):
                flow_lr = None
                prev_context = None
                if t > 0:
                    flow_lr = self._compute_flow(lr_seq[:, t - 1], lr_seq[:, t])
                    if prev_pred is not None:
                        if torch.rand(1).item() < teacher_p:
                            prev_context = target_seq[:, t - 1].detach()
                        else:
                            prev_context = prev_pred

                with autocast('cuda', enabled=self.use_amp):
                    pred = self.model(lr_seq[:, t], prev_hr=prev_context, flow=flow_lr)
                    self.loss_manager.compute(
                        pred, target_seq[:, t], self.current_epoch,
                        prev_pred=prev_pred, flow=flow_lr,
                    )
                    total_loss = total_loss + self.loss_manager.get_weighted_loss() / steps

                prev_pred = pred

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_sum += total_loss.detach()

            self.current_batch = batch_idx
            self.batch_logs = {
                'train_loss': total_loss.item(),
                'teacher_forcing_p': teacher_p,
                **{k: v.item() for k, v in self.loss_manager.losses.items()},
            }
            for cb in self.callbacks:
                cb.on_batch_end(self)

        return {
            'train_loss': (loss_sum / max(1, len(self.train_loader))).item(),
            'teacher_forcing_p': teacher_p,
        }

    def fit(self):
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self.train_logs = self._train_epoch()
            self.val_logs = {}
            lr = self.optimizer.param_groups[0]['lr']
            self.epoch_logs = {'lr': lr, **self.train_logs}
            for cb in self.callbacks:
                cb.on_epoch_end(self)
            self.epoch_logs.update(self.val_logs)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
