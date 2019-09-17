"""
Based on code in https://github.com/kimiyoung/ssl_bad_gan
"""

import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
import sys

img_dir = os.path.abspath(sys.argv[1])
base_label_dir = os.path.abspath(sys.argv[2])
train_set = torchvision.datasets.SVHN(img_dir, split="train", download=True)
print("SVHN: label_dir", base_label_dir)
print("SVHN: img_dir", img_dir)

#10 images per class
for per_class in [25, 50, 100]:
    label_dir = os.path.join(base_label_dir, str(10 * per_class)+"_balanced_labels")
    os.makedirs(label_dir, exist_ok=True)
    
    #generate 20 data splits
    for i in range(20):
        np.random.seed(i)
    
        indices = np.arange(len(train_set))
        np.random.shuffle(indices)
        mask = np.zeros(indices.shape[0], dtype=np.bool)
        labels = train_set.labels
        for j in range(10):
            mask[np.where(labels[indices] == j)[0][:per_class]] = True

        np.savez(os.path.join(label_dir, str(i)),
                 labeled_indices=indices[mask],
                 unlabeled_indices=indices[~mask])
