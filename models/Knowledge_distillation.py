#!/usr/bin/env python3
"""
Knowledge Distillation single-file trainer
- Runs on Ubuntu WSL (Linux) and uses GPU when available
- Single-file script with checkpointing (save & resume)
- Uses torch, torchvision

Usage examples:
# install (example using your 'uv' package manager):
# uv add torch torchvision torchaudio tqdm --index-url https://download.pytorch.org/whl/cu121

# Run training from scratch:
# python kd_train_uv_wsl.py --data-dir ./data --epochs 30 --batch-size 128

# Resume from checkpoint:
# python kd_train_uv_wsl.py --data-dir ./data --resume checkpoint.pth

"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ---------------------- Models ----------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ---------------------- Distillation loss ----------------------
def distillation_loss_fn(student_logits, teacher_logits, targets, T, alpha):
    # student_logits, teacher_logits: raw logits (no softmax)
    # T: temperature
    # alpha: weight for distillation loss
    ce = F.cross_entropy(student_logits, targets)
    # KL divergence between softened probabilities
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
    return alpha * kl + (1.0 - alpha) * ce, ce, kl

# ---------------------- Utilities ----------------------

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    if is_best:
        best_path = checkpoint_dir / 'model_best.pth'
        torch.save(state, best_path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

# ---------------------- Training & Validation ----------------------

def train_one_epoch(epoch, model_s, model_t, loader, optimizer, device, scaler, args):
    model_s.train()
    if model_t is not None:
        model_t.eval()

    running_loss = 0.0
    running_ce = 0.0
    running_kl = 0.0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Epoch {epoch}")
    for i, (images, targets) in pbar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            out_s = model_s(images)
            with torch.no_grad():
                out_t = model_t(images) if model_t is not None else None
            if out_t is not None:
                loss, ce, kl = distillation_loss_fn(out_s, out_t, targets, args.temperature, args.alpha)
            else:
                loss = F.cross_entropy(out_s, targets)
                ce = loss
                kl = torch.tensor(0.0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        running_ce += ce.item() * images.size(0)
        running_kl += (kl.item() if isinstance(kl, torch.Tensor) else float(kl)) * images.size(0)
        preds = out_s.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.0*correct/total})

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validate', leave=False)
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            out = model(images)
            loss = F.cross_entropy(out, targets)
            running_loss += loss.item() * images.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)
    return running_loss / total, 100.0 * correct / total

# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation - single file (WSL/GPU)')
    parser.add_argument('--data-dir', default='./data', help='data directory')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'imagenet-lite'], help='dataset')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--resume', default='', help='path to checkpoint to resume')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='where to save checkpoints')
    parser.add_argument('--teacher', default='resnet18', help='teacher model name (torchvision)')
    parser.add_argument('--student', default='smallcnn', help='student model: smallcnn or resnet18')
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--amp', action='store_true', help='use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)

    # device & WSL note
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Data
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
        valset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
    else:
        raise NotImplementedError('Only cifar10 implemented in single-file demo')

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Models
    if args.teacher.lower() == 'resnet18':
        teacher = models.resnet18(pretrained=True)
        # adapt final layer if number of classes differ
        if teacher.fc.out_features != num_classes:
            teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
    else:
        raise NotImplementedError('Only resnet18 supported as teacher in this demo')

    if args.student.lower() == 'smallcnn':
        student = SmallCNN(num_classes=num_classes)
    elif args.student.lower() == 'resnet18':
        student = models.resnet18(pretrained=False)
        if student.fc.out_features != num_classes:
            student.fc = nn.Linear(student.fc.in_features, num_classes)
    else:
        raise NotImplementedError('Unknown student model')

    teacher = teacher.to(device)
    student = student.to(device)

    # If GPU available, try to use multiple GPUs if present
    if torch.cuda.device_count() > 1:
        print('Multiple GPUs detected, wrapping models in DataParallel')
        teacher = nn.DataParallel(teacher)
        student = nn.DataParallel(student)

    # We don't train teacher here; assume teacher is pretrained or you provide checkpoint
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    best_acc = 0.0

    # Optionally resume
    if args.resume:
        print('Loading checkpoint:', args.resume)
        ckpt = load_checkpoint(args.resume, device)
        student.load_state_dict(ckpt['student_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt.get('scheduler_state', scheduler.state_dict()))
        start_epoch = ckpt.get('epoch', 0)
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, best_acc {best_acc}")

    # optionally allow loading a teacher checkpoint with a path 'teacher.pth' in the checkpoint dir
    teacher_ckpt_path = Path(args.checkpoint_dir) / 'teacher.pth'
    if teacher_ckpt_path.exists():
        print('Loading teacher checkpoint from', teacher_ckpt_path)
        ck = torch.load(teacher_ckpt_path, map_location=device)
        try:
            teacher.load_state_dict(ck['teacher_state'])
        except Exception:
            # if single gpu saved, handle DataParallel mismatch
            teacher.load_state_dict(ck)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(epoch+1, student, teacher, trainloader, optimizer, device, scaler, args)
        val_loss, val_acc = validate(student, valloader, device)
        scheduler.step()
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        checkpoint = {
            'epoch': epoch + 1,
            'student_state': student.module.state_dict() if hasattr(student, 'module') else student.state_dict(),
            'teacher_state': teacher.module.state_dict() if hasattr(teacher, 'module') else teacher.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        save_checkpoint(checkpoint, is_best, args.checkpoint_dir, filename=f'checkpoint_epoch_{epoch+1}.pth')

        print(f"Epoch {epoch+1}/{args.epochs} — train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}%, val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%, time: {time.time()-t0:.1f}s")

    print('Training finished. Best val acc: {:.2f}%'.format(best_acc))


if __name__ == '__main__':
    main()
