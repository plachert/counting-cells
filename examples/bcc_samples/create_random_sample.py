from src.data.datasets import BloodCellDataset
import pathlib
import numpy as np


ds = BloodCellDataset(pathlib.Path("/home/piotr/datasets/vision/blood_cell_count_and_types_detection"), is_train=False)
indices = np.random.choice(len(ds), size=30, replace=False)
images = []
for idx in indices:
    image, _ = ds[idx]
    images.append(image)

np.save("random_cell_images", np.array(images))
