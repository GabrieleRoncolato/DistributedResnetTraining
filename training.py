import os

import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torchvision.transforms import transforms

from resnet import resnet18

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)

    return dataloader


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print the progress
        progress = 100. * (batch_idx + 1) / len(train_loader)
        print(f'\rEpoch {epoch+1} [{batch_idx + 1}/{len(train_loader)}] - {progress:.2f}%', end='')

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    return train_loss, train_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    return val_loss, val_accuracy


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100. * correct / total

    return test_loss, test_accuracy


def save_model(model, val_loss, best_val_loss, path, device, rank):
    metrics_tensor = torch.tensor([val_loss]).to(device)

    # Gather metrics from all processes to rank 0
    if dist.get_rank() == 0:
        # Prepare a tensor to gather all metrics on rank 0
        gathered_metrics = [torch.zeros(1, device=device)] * dist.get_world_size()
    else:
        gathered_metrics = None

    dist.gather(metrics_tensor, gather_list=gathered_metrics, dst=0)
    gathered_metrics = torch.tensor(gathered_metrics, device=device)

    if dist.get_rank() == 0:
        # Assuming `gathered_metrics` contains the metrics from all processes
        current_best_metric, best_rank = torch.max(gathered_metrics, 0)
        current_best_metric = current_best_metric.item()

        # Check if the current best metric is an improvement
        if current_best_metric < best_val_loss:
            best_val_loss = current_best_metric
            # Inform other processes to proceed with saving the model
            proceed_with_save = torch.tensor([1], dtype=torch.int32, device=device)
        else:
            proceed_with_save = torch.tensor([0], dtype=torch.int32, device=device)
    else:
        proceed_with_save = torch.tensor([0], dtype=torch.int32, device=device)
        best_rank = torch.tensor([0], dtype=torch.int32, device=device)

    dist.broadcast(best_rank, src=0)
    best_rank = best_rank.item()

    # Broadcast the decision to save the model
    dist.broadcast(proceed_with_save, src=0)

    # The process with the best metric saves the model if improvement is detected
    if proceed_with_save.item() == 1 and dist.get_rank() == best_rank:
        torch.save(model.state_dict(), path)
        print(f"Best model saved in rank: {dist.get_rank()}")

    return best_val_loss


def main(rank, world_size):
    setup(rank, world_size)

    model = resnet18(3, 32, 32).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the path to your Tiny ImageNet dataset
    data_dir = '/home/ronco/Desktop/tiny-imagenet-200'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to the original Tiny ImageNet dimension
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the training and validation datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # Instantiate the DataLoaders
    train_loader = prepare(train_dataset, rank, world_size, batch_size=32, num_workers=1)
    val_loader = prepare(val_dataset, rank, world_size, batch_size=32, num_workers=1)
    test_loader = prepare(test_dataset, rank, world_size, batch_size=32, num_workers=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    num_epochs = 120

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        train_loss_tensor = torch.tensor([train_loss]).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_global_loss = train_loss_tensor.item() / dist.get_world_size()

        train_accuracy_tensor = torch.tensor([train_accuracy]).to(device)
        dist.all_reduce(train_accuracy_tensor, op=dist.ReduceOp.SUM)
        train_global_accuracy = train_accuracy_tensor.item() / dist.get_world_size()

        print(f'Train Epoch: {epoch} Loss: {train_global_loss:.6f} Accuracy: {train_global_accuracy:.2f}%')

        val_loss_tensor = torch.tensor([val_loss]).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_global_loss = val_loss_tensor.item() / dist.get_world_size()

        val_accuracy_tensor = torch.tensor([val_accuracy]).to(device)
        dist.all_reduce(val_accuracy_tensor, op=dist.ReduceOp.SUM)
        val_global_accuracy = val_accuracy_tensor.item() / dist.get_world_size()

        print(f'Validation Loss: {val_global_loss:.6f} Accuracy: {val_global_accuracy:.2f}%')

        # Save the best model
        best_val_loss = save_model(model, val_loss, best_val_loss, 'best_model.path', rank, device)

        train_losses.append(train_global_loss)
        val_losses.append(val_global_loss)
        train_accuracies.append(train_global_accuracy)
        val_accuracies.append(val_global_accuracy)

    test_loss, test_accuracy = test(model, test_loader, criterion, device)

    test_loss_tensor = torch.tensor([test_loss]).to(device)
    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
    test_global_loss = test_loss_tensor.item() / dist.get_world_size()

    test_accuracy_tensor = torch.tensor([test_accuracy]).to(device)
    dist.all_reduce(test_accuracy_tensor, op=dist.ReduceOp.SUM)
    test_global_accuracy = test_accuracy_tensor.item() / dist.get_world_size()

    print(f'Test Loss: {test_global_loss:.6f} Accuracy: {test_global_accuracy:.2f}%')

    # Plotting training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()

    plt.savefig('training_validation_plot.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    # suppose we have 1 gpus
    world_size = 1
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )
