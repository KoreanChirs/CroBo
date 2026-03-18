import sys
import math

from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None and log_writer.logger_type() == 'tensorboard':
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batches in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        multi_samples = batches[0]
        multi_samples = multi_samples.to(device, non_blocking=True)

        # [B, repeated_sampling, 2, C, H, W] -> [B*repeated_sampling, 2, C, H, W]
        multi_samples = multi_samples.reshape(-1, *multi_samples.shape[2:])
        # split into [global_crops, local_crops], each [B*repeated_sampling, C, H, W]
        list_frames = [v.squeeze(1) for v in multi_samples.chunk(2, dim=1)]

        with torch.cuda.amp.autocast():
            loss = model(
                list_imgs=list_frames, epoch=data_iter_step / len(data_loader) + epoch,
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and log_writer.logger_type() == 'tensorboard':
            log_writer.update(loss=loss_value_reduce, head="train_loss")
            log_writer.update(lr=lr, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
