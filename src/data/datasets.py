import pathlib
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Callable


class BloodCellDataset(Dataset):
    
    """https://www.kaggle.com/datasets/balakrishcodes/blood-cell-count-and-typesdetection"""
    
    def __init__(self, path: pathlib.Path, is_train: bool, transform: Callable | None = None):
        super().__init__()
        self.path = path
        self.train_or_valid = "train" if is_train else "valid" 
        self.labels = self._load_labels()
        self.image_paths = self._load_image_paths()
        self.transform = transform
        
        
    def _load_image_paths(self):
        paths = list(self.path.glob(f'images/{self.train_or_valid}/*.jpg'))
        return paths
        
        
    def _load_labels(self):
        def parse_line(line):
            target, x_center, y_center, w, h =  line.split()
            return int(target), float(x_center), float(y_center), float(w), float(h)
        labels = {}
        paths = list(self.path.glob(f'labels/{self.train_or_valid}/*.txt'))
        for path in paths:
            id_ = path.stem
            with open(path, "r") as file:
                lines = file.readlines()
            labels[id_] = [parse_line(line) for line in lines] 
        return labels
    
    def _read_image(self, image_path):
        image = np.array(Image.open(image_path))
        return image
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self._read_image(image_path)
        id_ = image_path.stem
        label = self.labels[id_]
        return image, label
        
        
if __name__ == "__main__":
    ds = BloodCellDataset(pathlib.Path("/home/piotr/datasets/vision/blood_cell_count_and_types_detection"), is_train=False)
    print(ds[0])
    
    
        
        