__all__ = ["ImageLabelDataset"]

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split 
from PIL import Image
import re

from classification_config import *

def sorted_alphanum(img_names):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda img_name: [convert(x) for x in re.split(r'([0-9]+)', img_name)]
    return sorted(img_names, key=alphanum_key)

class ImageLabelDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        self.main_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir)) 
        self.labels = pd.read_csv(label_path)     
        self.label_dict = dict(zip(self.labels['id'], self.labels['target'])) 

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_loc = os.path.join(self.main_dir, self.image_names[idx])
        image = Image.open(image_loc).convert('RGB')
        if self.transform is not None:
            tensor_img = self.transform(image)
        else:
            raise ValueError("transform parameter cannot be None！")
        label = self.label_dict[idx]
        return tensor_img, label

# 测试创建数据集
if __name__ == '__main__':
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])
    dataset = ImageLabelDataset(IMG_PATH, FASHION_LABELS_PATH, transform=transform)
    print(len(dataset))