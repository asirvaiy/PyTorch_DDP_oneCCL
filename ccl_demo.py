import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
import time
import torch.utils.data as Data
import torchvision
try:
   import intel_extension_for_pytorch as ipex
except:
   print("cant't import ipex")

import oneccl_bindings_for_pytorch


EPOCH = 1
LR = 0.001
DOWNLOAD = True
DATA = 'datasets/cifar10/'

def train(train_loader, net, criterion, optimizer, epoch, args):
    net.train()
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        t00 = time.time()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        t01 = time.time()
        if batch_idx % 1 == 0 or len(data) < args.batch_size:
            print('[{}] Train Epoch: {} [{:5d}/{} ({:6.2f}%)]\tLoss: {:.6f}\tdur: {:.5f}ms'.format(args.rank, epoch, args.world_size * batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx /             float(len(train_loader)), loss.item(), (t01-t00)*1000))
    t1 = time.time()
    print('time elapsed: {:.2f}s'.format(t1-t0))

def test(test_loader, net, criterion, optimizer, args):
    net.eval()
    test_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(args.device, non_blocking=True)
            if args.ipex and args.device == 'cpu':
                data = data.to(memory_format=torch.channels_last)
            target = target.to(args.device, non_blocking=True)
            if args.device == 'cpu':
                with torch.cpu.amp.autocast(enabled=args.bf16):
                    output = net(data)
                    test_loss += criterion(output, target).item() * len(data) # sum up batch loss
            else:
                output = net(data)
                loss = criterion(output, target)
                test_loss += criterion(output, target).item() * len(data) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += len(data)
    test_loss /= count
    print('[{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(args.rank, test_loss, correct, count, 100. * correct / count))
    return test_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch DistributedDataParallel Training')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--local_world_size', default=1, type=int, help='local world size')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--device', default='cpu', type=str, help='Device to run on, default to cpu')
    parser.add_argument('--ipex', action='store_true', help='with Intel(R) Extension for PyTorch*')
    parser.add_argument('--bf16', action='store_true', help='Train with BFloat16')
    parser.add_argument('--backend', default='gloo', type=str, help='DDP backend, default to gloo')
    parser.add_argument('--master_addr', default='127.0.0.1', type=str, help='Master Addr')
    parser.add_argument('--port', default='29500', type=str, help='Port')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    args = parser.parse_args()    
    
    

    mpi_world_size = int(os.environ.get('PMI_SIZE', -1))
    mpi_rank = int(os.environ.get('PMI_RANK', -1))
    if mpi_world_size > 0:
        os.environ['RANK'] = str(mpi_rank)
        os.environ['WORLD_SIZE'] = str(mpi_world_size)
    else:
        # set the default rank and world size to 0 and 1
        os.environ['RANK'] = str(os.environ.get('RANK', 0))
        os.environ['WORLD_SIZE'] = str(os.environ.get('WORLD_SIZE', 1))
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # your master address
    os.environ['MASTER_PORT'] = '29500'  # your master port
    # Initialize the process group with ccl backend
    dist.init_process_group(backend='ccl')
    
    device = 'cpu' #"xpu:{}".format(dist.get_rank())
    args.world_size = mpi_world_size
    args.rank = mpi_rank    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # train_dataset = torchvision.datasets.ImageFolder(
    #         root='{}/train'.format(DATA),
    #         transform=transform
    # )
    train_dataset = torchvision.datasets.CIFAR10(
            root=DATA,
            train=True,
            transform=transform,
            download=DOWNLOAD,
    )
    
    train_dataset_subset = torch.utils.data.Subset(train_dataset, range(4096))
    
    #test_dataset = torchvision.datasets.ImageNet(root='/home/krishna-intel/imagenet/data', split='val', transform=transform)
    #val_dataset_subset = torch.utils.data.Subset(test_dataset, range(1024))
    #calibration_data_loader = torch.utils.data.DataLoader(val_dataset_subset, batch_size=128, shuffle=False)


    sampler_train = None
    if mpi_world_size > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset_subset)
    train_loader = Data.DataLoader(
            dataset=train_dataset_subset,
            batch_size=args.batch_size,
            sampler=sampler_train
    )
    
    
    
    test_dataset = torchvision.datasets.CIFAR10(
            root=DATA,
            train=False,
            transform=transform,
            download=DOWNLOAD,
    )
    test_dataset_subset = torch.utils.data.Subset(test_dataset, range(1024))
    sampler_test = None
    if mpi_world_size > 1:
        sampler_test = torch.utils.data.distributed.DistributedSampler(test_dataset_subset)
    test_loader = Data.DataLoader(
            dataset=test_dataset_subset,
            batch_size=args.batch_size,
            sampler=sampler_test
    )

    net = torchvision.models.resnet18()
    net = net.to(args.device)
    lr_scaler = 1
    optimizer = torch.optim.SGD(net.parameters(), lr = LR * lr_scaler, momentum=0.9)

    net.train()
    if args.ipex:
        if args.device == 'cpu':
            if args.bf16 and args.backend == 'ccl':
                net, optimizer = ipex.optimize(net, optimizer=optimizer, dtype=torch.bfloat16, level="O1")
            else:
                net, optimizer = ipex.optimize(net, optimizer=optimizer, dtype=torch.float32, level="O1")

    if mpi_world_size > 1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=None)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    for epoch in range(EPOCH):
        train(train_loader, net, criterion, optimizer, epoch, args)
        if mpi_rank == 0 and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'checkpoint_{}.pth'.format(epoch))
        loss = test(test_loader, net, criterion, optimizer, args)
        if loss <= 0.000001:
            break        
