from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
from torchvision.datasets import ImageFolder


parser = argparse.ArgumentParser(description='faceexpression')

parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type= float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--train_path', default='./data/RaFD6/train/', type=str, help='train')
parser.add_argument('--test_path', default='./data/RaFD6/test/', type=str, help='test')
args = parser.parse_args()
print('learning rate is:', args.lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9
num_epochs = 250
batch_size = 50

transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomRotation(180),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=10, contrast=10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_dataset = ImageFolder(args.train_path, transform)
test_dataset = ImageFolder(args.test_path, transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

total_step=len(train_loader)
print('train_images', total_step)
model=models.resnext50_32x4d(pretrained=True).to(device)
checkpoint = torch.load('')
emotionnet.load_state_dict(checkpoint)

# Loss and optimizer
criterion  =nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr']=lr


# Train the model
use_cuda = torch.cuda.is_available()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

for epoch in range(num_epochs):
    # Decay learning rate
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = args.lr * decay_factor
        set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = args.lr
    print('learning_rate: %s' % str(current_lr))

    for i, data in enumerate(train_loader):

        images, labels= data
        #images, labels= Variable(images, requires_grad=True), Variable(labels,requires_grad=True)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs=model(images)
        loss=criterion(outputs, labels.long())


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % total_step == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
model.eval()
with torch.no_grad():
    correct=0
    total=0
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _, predicted=torch.max(outputs.data, 1)
        total+=labels.size(0)
        correct+=(predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint

torch.save(model.state_dict(), './models/trained_models/test_model0721.ckpt')










