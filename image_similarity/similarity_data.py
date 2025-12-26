__all__ = ["ImageDataset"]

import os
import torch
from torch.utils.data import Dataset

from PIL import Image
import re

from similarity_config import *

def sorted_alphanum(img_names):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda img_name: [convert(x) for x in re.split(r'([0-9]+)', img_name)]
    return sorted(img_names, key=alphanum_key)

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.main_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir))  

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_loc = os.path.join(self.main_dir, self.image_names[idx])
        image = Image.open(image_loc).convert('RGB')
        if self.transform is not None:
            tensor_img = self.transform(image)
        else:
            raise ValueError("transform parameters cannot be None！")
        return tensor_img, tensor_img

if __name__ == "__main__":
    # image_names = os.listdir(IMG_PATH)
    # print(image_names)
    # print(sorted_alphanum(image_names))
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])
    dataset = ImageDataset(IMG_PATH, transform)
    print(len(dataset))