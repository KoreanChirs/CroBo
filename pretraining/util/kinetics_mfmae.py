import os
import random
import pickle

from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


class CroBoCrop:
    def __init__(
        self,
        hflip_p=0.5,
        size=(224, 224),
        global_scale=(0.5, 1.0),
        local_scale=(0.3, 0.6),
        ratio=(3./4., 4./3.),
        interpolation=F.InterpolationMode.BICUBIC,
    ):
        self.hflip_p = hflip_p
        self.size = size
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, np_img):
        pil_img = F.to_pil_image(np_img)

        # Global crop (source view) from original frame
        i_g, j_g, h_g, w_g = transforms.RandomResizedCrop.get_params(
            pil_img, scale=self.global_scale, ratio=self.ratio
        )
        global_crop = F.resized_crop(pil_img, i_g, j_g, h_g, w_g,
                                     size=self.size, interpolation=self.interpolation)

        # Local crop (target view) further cropped from the global crop
        i_l, j_l, h_l, w_l = transforms.RandomResizedCrop.get_params(
            global_crop, scale=self.local_scale, ratio=self.ratio
        )
        local_crop = F.resized_crop(global_crop, i_l, j_l, h_l, w_l,
                                    size=self.size, interpolation=self.interpolation)

        # Consistent horizontal flip
        if random.random() < self.hflip_p:
            global_crop = F.hflip(global_crop)
            local_crop = F.hflip(local_crop)

        return global_crop, local_crop


class KineticsDataset(Dataset):
    def __init__(
        self,
        root,
        repeated_sampling=2,
    ):
        super().__init__()
        self.root = root
        with open(os.path.join(self.root, "labels", "label_1.0.pickle"), "rb") as f:
            self.samples = pickle.load(f)

        self.crop = CroBoCrop()
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.repeated_sampling = repeated_sampling

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = os.path.join(self.root, self.samples[index][1])
        vr = VideoReader(sample, num_threads=1, ctx=cpu(0))

        list_views = []
        for _ in range(self.repeated_sampling):
            frame = self.load_frame(vr)
            global_crop, local_crop = self.crop(frame)
            global_crop = self.basic_transform(global_crop)  # [C, H, W]
            local_crop = self.basic_transform(local_crop)    # [C, H, W]
            views = torch.stack([global_crop, local_crop], dim=0)  # [2, C, H, W]
            list_views.append(views)

        # [repeated_sampling, 2, C, H, W]
        # DataLoader will batch this to [B, repeated_sampling, 2, C, H, W]
        return torch.stack(list_views, dim=0), 0

    def load_frame(self, vr):
        idx = random.randint(0, len(vr) - 1)
        return vr[idx].asnumpy()
