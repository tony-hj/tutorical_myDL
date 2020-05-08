import numpy as np
import torch
import torchvision
import os


def get_loader(batch_size,input_size, num_workers):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    data_transforms = {'train':train_transform,'val':test_transform}
    
    data_dir = './dataset/'
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    train_loader = torch.utils.data.DataLoader(
        image_datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        image_datasets['val'],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    # test_loader = torch.utils.data.DataLoader(
        # image_datasets['test'],
        # batch_size=batch_size,
        # num_workers=num_workers,
        # shuffle=False,
        # pin_memory=True,
        # drop_last=False,
    # )
    dataloaders_dict = {'train':train_loader,'val':val_loader}
    
    return dataloaders_dict , image_datasets['train'].class_to_idx