import torch
import torchvision
import os
from flow_ssl.data.image_datasets import SVHN_


def make_sup_data_loaders(
        path, 
        batch_size, 
        num_workers, 
        transform_train, 
        transform_test, 
        use_validation=True, 
        val_size=5000, 
        shuffle_train=True,
        dataset="cifar10", 
        ):

    
    if dataset == "notmnist":
        test_set = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )
        return None, test_loader, 10

    if dataset.lower() == "svhn":
        ds = SVHN_
    else:
        ds = getattr(torchvision.datasets, dataset.upper())

    train_set = ds(root=path, train=True, download=True, transform=transform_train)
    num_classes = max(train_set.train_labels) + 1

    if use_validation:
        print("Using train (" + str(len(train_set.train_data)-val_size) + 
              ") + validation (" +str(val_size)+ ")")
        train_set.train_data = train_set.train_data[:-val_size]
        train_set.train_labels = train_set.train_labels[:-val_size]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.test_data = test_set.train_data[-val_size:]
        test_set.test_labels = test_set.train_labels[-val_size:]
        delattr(test_set, 'train_data')
        delattr(test_set, 'train_labels')
    else:
        test_set = ds(root=path, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            )
    test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

    return train_loader, test_loader, num_classes


