from __future__ import annotations

import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]


def list_image_names(root_dir: str, scale: int) -> list[str]:
    lr_dir = os.path.join(root_dir, f"input_{640 // scale}")
    hr_dir = os.path.join(root_dir, "target_640")
    lr_names = {n for n in os.listdir(lr_dir) if n.lower().endswith(".png")}
    hr_names = {n for n in os.listdir(hr_dir) if n.lower().endswith(".png")}
    return sorted(lr_names & hr_names)


def make_split(
    root_dir: str,
    scale: int,
    val_count: int = 120,
    seed: int = 42,
    train_all: bool = False,
) -> Split:
    names = list_image_names(root_dir, scale)
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    val = sorted(shuffled[:val_count])
    if train_all:
        train = names
    else:
        train = sorted(shuffled[val_count:])
    return Split(train=train, val=val)


class SRDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        scale: int = 2,
        patch_size: int = 96,
        names: list[str] | None = None,
        repeat: int = 16,
        augment: bool = True,
    ):
        self.root_dir = root_dir
        self.scale = scale
        self.lr_dir = os.path.join(root_dir, f"input_{640 // scale}")
        self.hr_dir = os.path.join(root_dir, "target_640")
        self.names = names if names is not None else list_image_names(root_dir, scale)
        self.patch_size = patch_size
        self.repeat = max(1, repeat)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names) * self.repeat

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx % len(self.names)]
        lr = cv2.imread(os.path.join(self.lr_dir, name), cv2.IMREAD_GRAYSCALE)
        hr = cv2.imread(os.path.join(self.hr_dir, name), cv2.IMREAD_GRAYSCALE)
        if lr is None or hr is None:
            raise FileNotFoundError(name)

        lr = lr.astype(np.float32) / 255.0
        hr = hr.astype(np.float32) / 255.0

        h, w = lr.shape
        ps = min(self.patch_size, h, w)
        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)
        lr_patch = lr[top : top + ps, left : left + ps]
        hr_patch = hr[
            top * self.scale : (top + ps) * self.scale,
            left * self.scale : (left + ps) * self.scale,
        ]

        if self.augment:
            lr_patch, hr_patch = paired_augment(lr_patch, hr_patch)

        lr_t = torch.from_numpy(np.ascontiguousarray(lr_patch)).float().unsqueeze(0)
        hr_t = torch.from_numpy(np.ascontiguousarray(hr_patch)).float().unsqueeze(0)
        return lr_t, hr_t


class SyntheticDegradeDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        scale: int = 2,
        patch_size: int = 96,
        names: list[str] | None = None,
        repeat: int = 16,
        augment: bool = True,
        sigma: float = 0.5,
        interp: str = "area",
    ):
        self.root_dir = root_dir
        self.scale = scale
        self.hr_dir = os.path.join(root_dir, "target_640")
        self.names = names if names is not None else list_image_names(root_dir, scale)
        self.patch_size = patch_size
        self.repeat = max(1, repeat)
        self.augment = augment
        self.sigma = sigma
        self.interp = {
            "area": cv2.INTER_AREA,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }[interp]

    def __len__(self) -> int:
        return len(self.names) * self.repeat

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx % len(self.names)]
        hr = cv2.imread(os.path.join(self.hr_dir, name), cv2.IMREAD_GRAYSCALE)
        if hr is None:
            raise FileNotFoundError(name)

        hr = hr.astype(np.float32)
        h, w = hr.shape
        ps_lr = min(self.patch_size, h // self.scale, w // self.scale)
        ps_hr = ps_lr * self.scale
        top = np.random.randint(0, h - ps_hr + 1)
        left = np.random.randint(0, w - ps_hr + 1)
        hr_patch = hr[top : top + ps_hr, left : left + ps_hr]
        degraded = hr_patch
        if self.sigma > 0:
            degraded = cv2.GaussianBlur(degraded, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        lr_patch = cv2.resize(degraded, (ps_lr, ps_lr), interpolation=self.interp)

        lr_patch = np.clip(lr_patch / 255.0, 0.0, 1.0)
        hr_patch = np.clip(hr_patch / 255.0, 0.0, 1.0)

        if self.augment:
            lr_patch, hr_patch = paired_augment(lr_patch, hr_patch)

        lr_t = torch.from_numpy(np.ascontiguousarray(lr_patch)).float().unsqueeze(0)
        hr_t = torch.from_numpy(np.ascontiguousarray(hr_patch)).float().unsqueeze(0)
        return lr_t, hr_t


def paired_augment(lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.random.random() < 0.5:
        lr = np.flip(lr, axis=1)
        hr = np.flip(hr, axis=1)
    if np.random.random() < 0.5:
        lr = np.flip(lr, axis=0)
        hr = np.flip(hr, axis=0)
    if np.random.random() < 0.5:
        k = np.random.randint(1, 4)
        lr = np.rot90(lr, k)
        hr = np.rot90(hr, k)
    return lr, hr


def load_pair(root_dir: str, name: str, scale: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lr_dir = os.path.join(root_dir, f"input_{640 // scale}")
    hr_dir = os.path.join(root_dir, "target_640")
    lr = cv2.imread(os.path.join(lr_dir, name), cv2.IMREAD_GRAYSCALE)
    hr = cv2.imread(os.path.join(hr_dir, name), cv2.IMREAD_GRAYSCALE)
    if lr is None or hr is None:
        raise FileNotFoundError(name)
    lr_t = torch.from_numpy(lr.astype(np.float32) / 255.0).view(1, 1, lr.shape[0], lr.shape[1]).to(device)
    hr_t = torch.from_numpy(hr.astype(np.float32) / 255.0).view(1, 1, hr.shape[0], hr.shape[1]).to(device)
    return lr_t, hr_t
