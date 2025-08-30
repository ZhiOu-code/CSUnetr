import os
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from utils.utils import resample_3d
from monai.data import decollate_batch
import torch.nn.functional as F
from medpy.metric.binary import assd  # pip install medpy


def compute_asd_per_class(pred, target):
    """
    仅计算指定类别的 ASD（平均表面距离）
    :param pred: (1, C, H, W, D)
    :param target: (1, C, H, W, D)
    :return: dict -> {13: val, 26: val}
    """
    selected_classes = [13, 26]
    asd_scores = {}
    for cls in selected_classes:
        pred_cls = pred[0, cls].cpu().numpy().astype(np.bool_)
        target_cls = target[0, cls].cpu().numpy().astype(np.bool_)
        if np.any(pred_cls) and np.any(target_cls):
            asd_val = assd(pred_cls, target_cls)
        else:
            asd_val = np.nan
        asd_scores[cls] = asd_val
    return asd_scores


def val_epoch(model, loader, epoch, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    start_time = time.time()
    all_asd_scores = []

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            with autocast(enabled=args.amp):
                logits = model_inferer(data) if model_inferer else model(data)

            target_shape = (256, 256, 256)

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(x) for x in val_labels_list]
            val_labels_convert = torch.stack(val_labels_convert).cuda(args.gpu)
            val_labels_convert = val_labels_convert.squeeze(0).argmax(dim=0)
            val_labels_convert = resample_3d(val_labels_convert.cpu().numpy(), target_shape)
            val_labels_convert = torch.tensor(val_labels_convert, device=args.gpu)

            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(x) for x in val_outputs_list]
            val_output_convert = torch.stack(val_output_convert).cuda(args.gpu)
            val_output_convert = val_output_convert.squeeze(0).argmax(dim=0)
            val_output_convert = resample_3d(val_output_convert.cpu().numpy(), target_shape)
            val_output_convert = torch.tensor(val_output_convert, device=args.gpu)

            val_labels_convert = F.one_hot(val_labels_convert.long(), num_classes=29).permute(3, 0, 1, 2).unsqueeze(0)
            val_output_convert = F.one_hot(val_output_convert.long(), num_classes=29).permute(3, 0, 1, 2).unsqueeze(0)

            asd_dict = compute_asd_per_class(val_output_convert, val_labels_convert)
            asd_scores = [asd_dict.get(13, np.nan), asd_dict.get(26, np.nan)]
            all_asd_scores.append(asd_scores)

            print(f"Val {epoch}/{args.max_epochs} {idx}/{len(loader)} time {time.time() - start_time:.2f}s")
            print("ASD (class 13 and 26) for this image:", asd_scores)
            start_time = time.time()

    mean_asd_scores = [round(np.nanmean([x[i] for x in all_asd_scores]), 4) for i in range(2)]

    print("\n==== Average ASD for class 13 and 26 ====")
    print(f"Class 13: {mean_asd_scores[0]:.4f}, Class 26: {mean_asd_scores[1]:.4f}")

    return mean_asd_scores


def run_validation(
    model,
    val_loader,
    args,
    model_inferer=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            val_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)

        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            mean_asd_scores = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                args=args,
                model_inferer=model_inferer,
                post_label=post_label,
                post_pred=post_pred,
            )
    return mean_asd_scores
