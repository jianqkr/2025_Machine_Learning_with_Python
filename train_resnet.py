import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item()

@dataclass
class TrainStats:
    loss: float
    acc: float

def build_transforms(img_size: int):
    weights = ResNet50_Weights.IMAGENET1K_V2
    mean, std = weights.transforms().mean, weights.transforms().std

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def param_groups_weight_decay(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]

def build_dmix_subset(
    dmix_root: str,
    train_tf,
    orig_class_to_idx: Dict[str, int],
    per_class: int,
    seed: int,
) -> Optional[Subset]:
    if not os.path.isdir(dmix_root):
        print(f"[DMIX] root not found: {dmix_root}")
        return None

    tmp = datasets.ImageFolder(dmix_root) 
    if len(tmp.samples) == 0:
        print(f"[DMIX] no images found under: {dmix_root}")
        return None

    dmix_to_orig: Dict[int, int] = {}
    for cname, dmix_idx in tmp.class_to_idx.items():
        if cname in orig_class_to_idx:
            dmix_to_orig[dmix_idx] = orig_class_to_idx[cname]

    def tgt_map(t: int) -> int:
        return dmix_to_orig.get(t, t)

    dmix_full = datasets.ImageFolder(dmix_root, transform=train_tf, target_transform=tgt_map)

    per_cls_indices: Dict[int, List[int]] = {}
    for i, (_, t) in enumerate(tmp.samples):
        if t in dmix_to_orig:
            per_cls_indices.setdefault(t, []).append(i)

    import random as _rnd
    _rnd.seed(seed)
    chosen: List[int] = []
    for t, idxs in per_cls_indices.items():
        k = per_class if len(idxs) > per_class else len(idxs)
        chosen.extend(_rnd.sample(idxs, k))

    if not chosen:
        print("[DMIX] no indices selected")
        return None

    print(f"[DMIX] selected {len(chosen)} images from {len(per_cls_indices)} classes (≈{per_class}/class)")
    return Subset(dmix_full, chosen)

def train_epoch(model, loader, optimizer, device, criterion) -> TrainStats:
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        b = images.size(0)
        total_loss += loss.item() * b
        total_acc  += accuracy(outputs, targets) * b
        total_n    += b

    return TrainStats(loss=total_loss/total_n, acc=total_acc/total_n)

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total, correct = 0, 0
    cls_total = torch.zeros(num_classes, dtype=torch.long)
    cls_corr  = torch.zeros(num_classes, dtype=torch.long)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits  = model(images)
        pred    = logits.argmax(dim=1)

        total   += targets.size(0)
        correct += (pred == targets).sum().item()

        for t, p in zip(targets.view(-1), pred.view(-1)):
            cls_total[t] += 1
            if p == t:
                cls_corr[t] += 1

    top1 = correct / total if total else 0.0
    per_class = {f"class_{c}": (cls_corr[c].item()/cls_total[c].item())
                 for c in range(num_classes) if cls_total[c] > 0}
    return top1, per_class

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 with DiffuseMix")
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dmix-root', type=str, required=True)
    parser.add_argument('--dmix-per-class', type=int, default=15, help='images per class from DiffuseMix to add to TRAIN')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup-epochs', type=int, default=5, help='AdamW + cosine')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dir = os.path.join(args.data_root, 'train')
    val_dir   = os.path.join(args.data_root, 'val')

    probe = datasets.ImageFolder(train_dir)
    classes = probe.classes
    num_classes = len(classes)

    train_tf, eval_tf = build_transforms(args.img_size)

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    dmix_subset = build_dmix_subset(
        args.dmix_root, train_tf, train_set.class_to_idx, args.dmix_per_class, args.seed
    )
    if dmix_subset is not None:
        train_set = ConcatDataset([train_set, dmix_subset])
    else:
        print("[DMIX] no DiffuseMix data added.")

    val_set  = datasets.ImageFolder(val_dir,  transform=eval_tf)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    print(f"train: {sum(len(ds) for ds in train_set.datasets) if isinstance(train_set, ConcatDataset) else len(train_set)} | "
          f"val: {len(val_set)}")

    model = build_model(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    base_bsz = 64
    scaled_lr = args.lr * (args.batch_size / base_bsz)
    optimizer = optim.AdamW(param_groups_weight_decay(model, args.wd),
                            lr=scaled_lr, betas=(0.9, 0.999))

    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=max(1, args.warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[args.warmup_epochs])

    # Train
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f'best_resnet50_dmix_{args.seed}.pth')
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, device, criterion)
        val_acc, _ = evaluate(model, val_loader, device, num_classes)
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{args.epochs} | lr={current_lr:.2e} "
              f"| train_loss={tr.loss:.4f} train_acc={tr.acc:.4f} "
              f"val_acc={val_acc:.4f} time={elapsed:.1f}s")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model': model.state_dict(),
                        'classes': classes,
                        'img_size': args.img_size}, ckpt_path)

    print(f"Best val acc: {best_val:.4f} | ckpt: {ckpt_path}")

if __name__ == '__main__':
    main()
