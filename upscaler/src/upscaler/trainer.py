from pathlib import Path

import torch
from torch.amp import autocast, GradScaler

from src.common.schedule import scheduled_value
from src.losses.losses import CharbonnierLoss, VGGPerceptualLoss, TemporalConsistencyLoss, SobelEdgeLoss, FFTLoss
from src.utils.fast_flow import backward_warp, resize_flow


class UpscalerTrainer:
    def __init__(self, model, optimizer, lr_scheduler, train_loader, val_loader, device, max_epochs,
                 callbacks=None, use_amp=True, grad_clip=1.0,
                 loss_weights=None, loss_schedule=None, teacher_forcing=None,
                 rollout_length=None, flow_estimator=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.callbacks = callbacks or []
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.grad_clip = grad_clip

        self.base_loss_weights = loss_weights or {}
        self.loss_schedule = loss_schedule or []
        self.teacher_forcing = teacher_forcing or {
            'start_p': 0.0, 'end_p': 0.0, 'ramp_epochs': 0, 'mode': 'linear', 'delay_epochs': 0,
        }
        self.rollout_length = rollout_length
        self.flow_estimator = flow_estimator

        self.charbonnier = CharbonnierLoss().to(device)
        self.perceptual = VGGPerceptualLoss().to(device)
        self.temporal = TemporalConsistencyLoss().to(device)
        self.edge_loss = SobelEdgeLoss().to(device)
        self.fft_loss = FFTLoss().to(device)

        self.current_epoch = -1
        self.current_batch = -1
        self.batch_logs = {}
        self.train_logs = {}
        self.val_logs = {}
        self.epoch_logs = {}

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
        }
        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        torch.save(state, path)

    def _get_loss_weights(self, epoch):
        weights = dict(self.base_loss_weights)
        for stage in self.loss_schedule:
            if epoch >= int(stage.get('from_epoch', 0)):
                weights.update(stage.get('weights', {}))
        return weights

    def _teacher_forcing_p(self, epoch):
        delay = int(self.teacher_forcing.get('delay_epochs', 0))
        if epoch < delay:
            return 0.0
        return scheduled_value(
            epoch - delay,
            float(self.teacher_forcing.get('start_p', 0.0)),
            float(self.teacher_forcing.get('end_p', 0.0)),
            int(self.teacher_forcing.get('ramp_epochs', 0)),
            str(self.teacher_forcing.get('mode', 'linear')),
        )

    @torch.no_grad()
    def _compute_flow(self, lr_prev, lr_curr):
        if self.flow_estimator is None:
            return None
        return self.flow_estimator.calc_batch(lr_prev, lr_curr)

    def _train_epoch(self):
        self.model.train()
        loss_sum = 0.0
        teacher_p = self._teacher_forcing_p(self.current_epoch)
        lw = self._get_loss_weights(self.current_epoch)
        w_charb = float(lw.get('charbonnier', 1.0))
        w_perc = float(lw.get('perceptual', 0.0))
        w_temp = float(lw.get('temporal', 0.0))
        w_edge = float(lw.get('edge', 0.0))
        w_fft = float(lw.get('fft', 0.0))

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
                    gt = target_seq[:, t]
                    loss = w_charb * self.charbonnier(pred, gt)
                    if w_perc > 0:
                        loss = loss + w_perc * self.perceptual(pred, gt)
                    if w_edge > 0:
                        loss = loss + w_edge * self.edge_loss(pred, gt)
                    if w_fft > 0:
                        loss = loss + w_fft * self.fft_loss(pred, gt)
                    if w_temp > 0 and prev_pred is not None and flow_lr is not None:
                        flow_hr = resize_flow(flow_lr, pred.shape[-2], pred.shape[-1])
                        loss = loss + w_temp * self.temporal(pred, backward_warp(prev_pred.detach(), flow_hr))

                total_loss = total_loss + loss / steps
                prev_pred = pred

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_sum += total_loss

            self.current_batch = batch_idx
            self.batch_logs = {'train_loss': total_loss, 'teacher_forcing_p': teacher_p}
            for cb in self.callbacks:
                cb.on_batch_end(self)

        return {'train_loss': loss_sum / max(1, len(self.train_loader)), 'teacher_forcing_p': teacher_p}

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        loss_sum = 0.0
        visuals = []
        for lr_seq, target_seq in self.val_loader:
            lr_seq = lr_seq.to(self.device, non_blocking=True)
            target_seq = target_seq.to(self.device, non_blocking=True)
            steps = min(lr_seq.shape[1], self.rollout_length or lr_seq.shape[1])
            prev_pred = None
            batch_loss = 0.0
            visual_t = min(1, steps - 1)
            with autocast('cuda', enabled=self.use_amp):
                for t in range(steps):
                    flow_lr = self._compute_flow(lr_seq[:, t - 1], lr_seq[:, t]) if t > 0 else None
                    pred = self.model(lr_seq[:, t], prev_hr=prev_pred, flow=flow_lr)
                    batch_loss += self.charbonnier(pred, target_seq[:, t]).item()
                    if len(visuals) < 3 and t == visual_t:
                        visuals.append({
                            'current_lr': lr_seq[0, t].cpu(),
                            'current_upscale': pred[0].cpu(),
                            'current_true': target_seq[0, t].cpu(),
                        })
                    prev_pred = pred
            loss_sum += batch_loss / steps
        return {'val_loss': loss_sum / max(1, len(self.val_loader)), 'visuals': visuals}

    def fit(self):
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            self.train_logs = self._train_epoch()
            self.val_logs = self._validate()
            lr = self.optimizer.param_groups[0]['lr']
            self.epoch_logs = {'lr': lr, **self.train_logs, **self.val_logs}
            for cb in self.callbacks:
                cb.on_epoch_end(self)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
