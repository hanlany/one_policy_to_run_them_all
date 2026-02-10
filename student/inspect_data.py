import numpy as np
import os

file_path = 'teacher_dataset.npz'
if not os.path.exists(file_path):
    print(f"File {file_path} not found.")
else:
    data = np.load(file_path)
    print("Keys in the npz file:", data.files)
    for key in data.files:
        print(f"Key: {key}, Shape: {data[key].shape}, Dtype: {data[key].dtype}")
