
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=config.data_root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=config.data_root,
        train=False,
        download=True,
        transform=transform
    )

    pin_memory = config.device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory
    )

    return train_loader, test_loader
