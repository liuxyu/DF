from model import DF, AWF
from data import WFDataset, load_data
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

NB_CLASSES = 101
EPOCH = 30
BATCH_SIZE = 128
LR = 0.001

full_data = WFDataset("./data/tor_100w_2500tr.npz")
train_split= 0.8
validate_split = 0.15
test_split = 0.05
shuffle_dataset = True
random_seed = 16
dataset_size = len(full_data)
indices = list(range(dataset_size))
train_size = int(train_split * dataset_size)
validation_size = int(validate_split * dataset_size)
test_size = int(dataset_size - train_size - validation_size)
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices, test_indices= indices[:train_size], indices[train_size:train_size+validation_size], indices[train_size+validation_size:]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(full_data, batch_size=BATCH_SIZE, 
                                           sampler=train_sampler)
validation_loader = DataLoader(full_data, batch_size=BATCH_SIZE,
                                            sampler=valid_sampler)
test_loader = DataLoader(full_data, batch_size=BATCH_SIZE,
                                            sampler=test_sampler)

cuda_gpu = torch.cuda.is_available()
cnn = AWF(NB_CLASSES).float()
if(cuda_gpu):
    cnn = torch.nn.DataParallel(cnn, device_ids=[0]).cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.cuda()
        b_y = b_y.cuda()
        output = cnn(b_x.float())[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            corrects = 0
            avg_loss = 0
            for _, (b_x, b_y) in enumerate(validation_loader):
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                logit = cnn(b_x.float())[0]
                loss = loss_func(logit, b_y)
                avg_loss += loss.item()
                corrects += (torch.max(logit, 1)
                            [1].view(b_y.size()).data == b_y.data).sum()
            
            size = validation_size
            avg_loss /= size
            accuracy = 100.0 * corrects/size
            print('Epoch: {:2d}({:6d}/{}) Evaluation - loss: {:.6f}  acc: {:3.4f}%({}/{})'.format(
                                                                            epoch,
                                                                            step * 128,
                                                                            dataset_size,
                                                                            avg_loss, 
                                                                            accuracy, 
                                                                            corrects, 
                                                                            size))

for _, (b_x, b_y) in enumerate(test_loader):
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                logit = cnn(b_x.float())[0]
                loss = loss_func(logit, b_y)
                avg_loss += loss.item()
                corrects += (torch.max(logit, 1)
                            [1].view(b_y.size()).data == b_y.data).sum()
            
size = test_size
accuracy = 100.0 * corrects/size
print(accuracy)