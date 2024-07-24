from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import os

class Custom_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB") # X데이터
        label = self.data_frame.iloc[idx, 2]        # Y데이터

        if self.transform:
            image = self.transform(image)
            
        return image, label

    def _find_classes(self):
        class_names = self.data_frame.iloc[:, 1].unique().tolist()
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        return class_names, class_to_idx