"""Train one frozen R1 ROI candidate without touching locked-test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def emit(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class R1Dataset(Dataset):
    def __init__(self, root: Path, split: str, size: int, augment: bool):
        self.root = root
        manifest = root / "ROI_R1_EXECUTION_MANIFEST.jsonl"
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        self.rows = [row for row in self.rows if row["split"] == split]
        if not self.rows:
            raise RuntimeError(f"EMPTY_FROZEN_SPLIT:{split}")
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        with Image.open(self.root / row["mask_path"]) as source:
            mask = ImageOps.grayscale(source).resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment:
            rng = random.random
            if rng() < 0.5:
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)
            if rng() < 0.6:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            if rng() < 0.6:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
            if rng() < 0.5:
                image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
        mask_array = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask_array[None, ...].copy())
        return image_tensor, mask_tensor, row["canonical_source_id"], row["image_sha256"]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = (32, 64, 128, 256)
        self.encoders = nn.ModuleList()
        previous = 3
        for width in widths:
            self.encoders.append(DoubleConv(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.center = DoubleConv(256, 512)
        self.decoders = nn.ModuleList([DoubleConv(512 + 256, 256), DoubleConv(256 + 128, 128), DoubleConv(128 + 64, 64), DoubleConv(64 + 32, 32)])
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.center(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.out(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.layers = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((x, skip), dim=1))


class ResNet18UNet(nn.Module):
    def __init__(self, pretrained_path: Path):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(pretrained)
        self.up3 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up1 = Up(128, 64, 64)
        self.up0 = Up(64, 64, 64)
        self.out = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        enc = self.encoder
        a = enc.relu(enc.bn1(enc.conv1(x)))
        b = enc.layer1(enc.maxpool(a))
        c = enc.layer2(b)
        d = enc.layer3(c)
        e = enc.layer4(d)
        x = self.up3(e, d)
        x = self.up2(x, c)
        x = self.up1(x, b)
        x = self.up0(x, a)
        x = F.interpolate(x, size=(x.shape[-2] * 2, x.shape[-1] * 2), mode="bilinear", align_corners=False)
        return self.out(x)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = logits.sigmoid()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1) / (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1)
    return bce + 1 - dice.mean()


def boundary_f1(pred: torch.Tensor, truth: torch.Tensor, tolerance: int = 2) -> float:
    pred = pred.float()[None, None]
    truth = truth.float()[None, None]
    kernel = tolerance * 2 + 1
    pred_eroded = 1 - F.max_pool2d(1 - pred, kernel, stride=1, padding=tolerance)
    truth_eroded = 1 - F.max_pool2d(1 - truth, kernel, stride=1, padding=tolerance)
    pred_boundary = pred.bool() & ~pred_eroded.bool()
    truth_boundary = truth.bool() & ~truth_eroded.bool()
    pred_dilated = F.max_pool2d(pred_boundary.float(), kernel, stride=1, padding=tolerance).bool()
    truth_dilated = F.max_pool2d(truth_boundary.float(), kernel, stride=1, padding=tolerance).bool()
    precision = (pred_boundary & truth_dilated).sum().item() / max(pred_boundary.sum().item(), 1)
    recall = (truth_boundary & pred_dilated).sum().item() / max(truth_boundary.sum().item(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def evaluate(model, loader, device: torch.device, output: Path, save_masks: bool) -> dict:
    model.eval()
    if save_masks:
        output.mkdir(parents=True, exist_ok=True)
    cases = []
    with torch.inference_mode():
        for images, targets, ids, hashes in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            probabilities = logits.sigmoid()
            predictions = probabilities >= 0.5
            for index, (case_id, image_sha) in enumerate(zip(ids, hashes)):
                pred = predictions[index, 0]
                truth = targets[index, 0] >= 0.5
                intersection = int((pred & truth).sum().item())
                pred_count = int(pred.sum().item())
                truth_count = int(truth.sum().item())
                dice = (2 * intersection + 1) / (pred_count + truth_count + 1)
                iou = (intersection + 1) / (pred_count + truth_count - intersection + 1)
                b_f1 = boundary_f1(pred.cpu(), truth.cpu(), 2)
                if save_masks:
                    mask = (pred.cpu().numpy().astype(np.uint8) * 255)
                    Image.fromarray(mask, mode="L").save(output / f"{image_sha}.png")
                cases.append({"canonical_source_id": case_id, "image_sha256": image_sha, "dice": float(dice), "iou": float(iou), "boundary_f1_2px": b_f1, "prediction_positive_fraction": pred_count / pred.numel(), "truth_positive_fraction": truth_count / truth.numel()})
    dices = [row["dice"] for row in cases]
    ious = [row["iou"] for row in cases]
    boundary = [row["boundary_f1_2px"] for row in cases]
    return {"count": len(cases), "mean_dice": float(np.mean(dices)), "median_dice": float(np.median(dices)), "mean_iou": float(np.mean(ious)), "median_iou": float(np.median(ious)), "mean_boundary_f1_2px": float(np.mean(boundary)), "dice_lt_0_80_count": sum(value < 0.80 for value in dices), "worst_20": sorted(cases, key=lambda row: row["dice"])[:20], "cases": cases}


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("small_unet", "resnet18_unet"), required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrained-resnet18", type=Path)
    parser.add_argument("--training-code-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--pilot-steps", type=int, default=100)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    if args.effective_batch % args.microbatch:
        raise ValueError("EFFECTIVE_BATCH_NOT_DIVISIBLE_BY_MICROBATCH")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("DGX_CUDA_REQUIRED_FOR_FORMAL_ROI_TRAINING")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    train_set = R1Dataset(args.data_root, "TRAIN", args.image_size, True)
    dev_set = R1Dataset(args.data_root, "DEV", args.image_size, False)
    if len(train_set) != 1658 or len(dev_set) != 384:
        raise RuntimeError(f"R1_SUPPORT_MISMATCH:{len(train_set)}:{len(dev_set)}")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.microbatch, shuffle=True, generator=generator, num_workers=4, pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev_set, batch_size=args.microbatch, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    if args.model == "small_unet":
        model = SmallUNet()
    else:
        if not args.pretrained_resnet18 or not args.pretrained_resnet18.is_file():
            raise RuntimeError("RESNET18_PRETRAINED_WEIGHTS_MISSING")
        model = ResNet18UNet(args.pretrained_resnet18)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    accumulate = args.effective_batch // args.microbatch
    steps_per_epoch = int(np.ceil(len(train_set) / args.microbatch))
    total_steps = steps_per_epoch * args.epochs
    global_step = 0
    pilot_elapsed = None
    pilot_peak = None
    history_path = args.output / "epoch_history.jsonl"
    if history_path.exists():
        raise RuntimeError("REFUSE_NONEMPTY_RUN_DIRECTORY")
    best_dice = -1.0
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, targets, _, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NONFINITE_LOSS_EPOCH_{epoch}_STEP_{step}")
            (loss / accumulate).backward()
            epoch_loss_sum += float(loss.item()) * len(images)
            if step % accumulate == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step == args.pilot_steps:
                torch.cuda.synchronize()
                pilot_elapsed = time.monotonic() - started
                pilot_peak = torch.cuda.max_memory_allocated(device)
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                estimate_hours = pilot_elapsed / max(global_step, 1) * total_steps / 3600
                emit(args.output / "resource_trace.jsonl", {"event": "100_STEP_PILOT", "steps": global_step, "elapsed_seconds": pilot_elapsed, "seconds_per_step": pilot_elapsed / global_step, "projected_training_hours": estimate_hours, "cuda_peak_allocated_bytes": pilot_peak, "cuda_free_bytes": free_bytes, "cuda_total_bytes": total_bytes, "effective_batch": args.effective_batch, "microbatch": args.microbatch})
                torch.cuda.reset_peak_memory_stats(device)
                if args.pilot_only:
                    pilot_summary = {"event": "PILOT_COMPLETE", "model_type": args.model, "steps": global_step, "elapsed_seconds": pilot_elapsed, "seconds_per_step": pilot_elapsed / global_step, "projected_full_training_hours": estimate_hours, "cuda_peak_allocated_bytes": pilot_peak, "cuda_free_bytes_after_pilot": free_bytes, "cuda_total_bytes": total_bytes, "effective_batch": args.effective_batch, "microbatch": args.microbatch, "manifest_sha256": args.manifest_sha256, "training_code_sha256": args.training_code_sha256}
                    emit(args.output / "resource_trace.jsonl", pilot_summary)
                    print(json.dumps(pilot_summary, sort_keys=True), flush=True)
                    return
            if global_step == 1:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                emit(args.output / "resource_trace.jsonl", {"event": "TRAIN_START", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cuda_name": torch.cuda.get_device_name(device), "cuda_free_bytes": free_bytes, "cuda_total_bytes": total_bytes, "code_sha256": args.training_code_sha256, "manifest_sha256": args.manifest_sha256})
        dev_metrics = evaluate(model, dev_loader, device, args.output / "dev_predictions", False)
        improved = dev_metrics["mean_dice"] > best_dice + 1e-6
        record = {"epoch": epoch, "train_loss": epoch_loss_sum / len(train_set), "dev_mean_dice": dev_metrics["mean_dice"], "dev_mean_iou": dev_metrics["mean_iou"], "dev_mean_boundary_f1_2px": dev_metrics["mean_boundary_f1_2px"], "dev_dice_lt_0_80_count": dev_metrics["dice_lt_0_80_count"], "best_checkpoint": improved, "elapsed_seconds": time.monotonic() - started, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
        emit(history_path, record)
        emit(args.output / "dev_per_image_metrics.jsonl", {"epoch": epoch, **dev_metrics})
        state = {"schema_version": "ROI_R1_CHECKPOINT_V1", "model_type": args.model, "epoch": epoch, "seed": args.seed, "image_size": args.image_size, "effective_batch": args.effective_batch, "microbatch": args.microbatch, "manifest_sha256": args.manifest_sha256, "training_code_sha256": args.training_code_sha256, "pretrained_resnet18_sha256": sha256_file(args.pretrained_resnet18) if args.pretrained_resnet18 else None, "model_state_dict": cpu_state(model), "dev_mean_dice": dev_metrics["mean_dice"]}
        torch.save(state, args.output / "last.pt")
        if improved:
            best_dice = dev_metrics["mean_dice"]
            best_epoch = epoch
            torch.save(state, args.output / "best.pt")
        print(json.dumps(record, sort_keys=True), flush=True)
    best_state = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    best_metrics = evaluate(model, dev_loader, device, args.output / "best_dev_predictions", True)
    (args.output / "best_dev_metrics.json").write_text(json.dumps(best_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total_elapsed = time.monotonic() - started
    torch.cuda.synchronize()
    summary = {"status": "PASS_FINITE_COMPLETE", "model_type": args.model, "epochs_completed": args.epochs, "best_epoch": best_epoch, "best_dev_dice": best_dice, "best_dev_iou": best_metrics["mean_iou"], "best_dev_boundary_f1_2px": best_metrics["mean_boundary_f1_2px"], "best_dev_dice_lt_0_80_count": best_metrics["dice_lt_0_80_count"], "pilot_elapsed_seconds": pilot_elapsed, "pilot_peak_cuda_allocated_bytes": pilot_peak, "training_elapsed_seconds": total_elapsed, "gpu_hours": total_elapsed / 3600, "maximum_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device), "checkpoint_sha256": {name: sha256_file(args.output / name) for name in ("best.pt", "last.pt")}, "dev_metrics_path": str(args.output / "dev_per_image_metrics.jsonl"), "best_dev_metrics_sha256": sha256_file(args.output / "best_dev_metrics.json"), "manifest_sha256": args.manifest_sha256, "training_code_sha256": args.training_code_sha256}
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    emit(args.output / "resource_trace.jsonl", {"event": "TRAIN_COMPLETE", **summary})
    print(json.dumps({"event": "TRAIN_COMPLETE", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
