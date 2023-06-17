import torch as t
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as tf
import matplotlib.pyplot as plt
import datetime
from torch.utils.tensorboard import SummaryWriter

#variables
batch_size = 2

#transform all images to same size
transform = tv.transforms.Compose(
    [tv.transforms.ToTensor()])

#load data from folder
#training data
dataset_training = tv.datasets.ImageFolder(root='.\Pictures', transform=transform)
#dataset_training = tv.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
dataloader_training = t.utils.data.DataLoader(dataset_training, shuffle = True)

#validation data
#validation_set = tv.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
#validation_loader = t.utils.data.DataLoader(validation_set, batch_size, shuffle=False)

# PyTorch models inherit from torch.nn.Module
class CarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        print(self)
        self.fc1 = nn.Linear(1 * 64 * 64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(tf.relu(self.conv1(x)))
        x = self.pool(tf.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64)
        x = tf.relu(self.fc1(x))
        x = tf.relu(self.fc2(x))
        x = self.fc3(x)
        return x    

model = CarClassifier()

loss_fn = nn.CrossEntropyLoss()

# Optimizers specified in the torch.optim package
optimizer = t.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(dataloader_training):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader_training) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/custom_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 2

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with t.no_grad():
        for i, vdata in enumerate(dataloader_training):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        t.save(model.state_dict(), model_path)

    epoch_number += 1