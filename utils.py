import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100

def show_images(images):
    grid_img = torchvision.utils.make_grid(images, nrow=3)
    plt.imshow(grid_img.permute(1,2,0))
    plt.show()

def prepare_dataloader(
    num_workers=8, train_batch_size=128, eval_batch_size=256, dataset=CIFAR10
):

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_set = dataset(
        root="data", train=True, download=True, transform=train_transform
    )

    test_set = dataset(
        root="data", train=False, download=True, transform=test_transform
    )

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=eval_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
    )

    return train_loader, test_loader