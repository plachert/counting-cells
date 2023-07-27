import pathlib
from torch.utils.data import Dataset


class BloodCellDataset:
    
    """https://www.kaggle.com/datasets/balakrishcodes/blood-cell-count-and-typesdetection"""
    
    def __init__(self, path: pathlib.Path, is_train: bool):
        self.path = path
        self.train_or_valid = "train" if is_train else "valid" 
        self.labels = self._load_labels()
        
        
    def _load_labels(self):
        def parse_line(line):
            target, x_center, y_center, w, h =  line.split()
            return int(target), float(x_center), float(y_center), float(w), float(h)
        labels = {}
        paths = list(self.path.glob(f'labels/labels/{self.train_or_valid}/*.txt'))
        for path in paths:
            id_ = path.stem
            with open(path, "r") as file:
                lines = file.readlines()
            labels[id_] = [parse_line(line) for line in lines] 
        return labels
        
        
if __name__ == "__main__":
    ds = BloodCellDataset(pathlib.Path("/home/piotr/datasets/vision/blood_cell_count_and_types_detection"), is_train=False)
    print(ds.labels)
    
    
        
        