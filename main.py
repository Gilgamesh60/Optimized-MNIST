import torch 
import torch.nn as nn
import torch.nn.functional as F
class ModelM3(nn.Module):
    def __init__(self):
        super(ModelM3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, bias=False)       # output becomes 26x26
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, 3, bias=False)      # output becomes 24x24
        self.conv2_bn = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 64, 3, bias=False)      # output becomes 22x22
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 80, 3, bias=False)      # output becomes 20x20
        self.conv4_bn = nn.BatchNorm2d(80)
        self.conv5 = nn.Conv2d(80, 96, 3, bias=False)      # output becomes 18x18
        self.conv5_bn = nn.BatchNorm2d(96)
        self.conv6 = nn.Conv2d(96, 112, 3, bias=False)     # output becomes 16x16
        self.conv6_bn = nn.BatchNorm2d(112)
        self.conv7 = nn.Conv2d(112, 128, 3, bias=False)    # output becomes 14x14
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 144, 3, bias=False)    # output becomes 12x12
        self.conv8_bn = nn.BatchNorm2d(144)
        self.conv9 = nn.Conv2d(144, 160, 3, bias=False)    # output becomes 10x10
        self.conv9_bn = nn.BatchNorm2d(160)
        self.conv10 = nn.Conv2d(160, 176, 3, bias=False)   # output becomes 8x8
        self.conv10_bn = nn.BatchNorm2d(176)
        #dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5,step=0.1)
        dropout_rate =  0.30000000000000004
        self.drop1=nn.Dropout2d(p=dropout_rate)   
        #fc2_input_dim =  trial.suggest_int("fc2_input_dim", 32, 128,32)
        fc2_input_dim =96
        self.fc1 = nn.Linear( 11264,fc2_input_dim)
        #dropout_rate2 = trial.suggest_float("dropout_rate2", 0, 0.3,step=0.1)
        dropout_rate2=0.2
        self.drop2=nn.Dropout2d(p=dropout_rate2)
        self.fc2 = nn.Linear(fc2_input_dim, 10)
        self.fc1_bn = nn.BatchNorm1d(10)
    def forward(self, x):
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        conv6 = F.relu(self.conv6_bn(self.conv6(conv5)))
        conv7 = F.relu(self.conv7_bn(self.conv7(conv6)))
        conv8 = F.relu(self.conv8_bn(self.conv8(conv7)))
        conv9 = F.relu(self.conv9_bn(self.conv9(conv8)))
        conv10 = F.relu(self.conv10_bn(self.conv10(conv9)))
        x=  self.drop1(conv10)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x




def train(EPOCHS,optimizer,model,trainloader,testloader,DEVICE):
  n_total_steps = len(trainloader) 
  criterion=nn.CrossEntropyLoss()
  for epoch in range(EPOCHS):
        model.train()
       
        for batch_idx, (images, labels) in enumerate(trainloader):

              images, labels = images.to(DEVICE), labels.to(DEVICE)

              optimizer.zero_grad()
              output = model(images)
              loss = criterion(output, labels)
              loss.backward()
              optimizer.step()
              if (batch_idx+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{n_total_steps}],Loss: {loss.item():.4f}')
                
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testloader):
                  # Limiting validation images.
                # if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                  #    break
                  images, labels = images.to(DEVICE), labels.to(DEVICE)
                  output = model(images)
                  # Get the index of the max log-probability.
                  pred = output.argmax(dim=1, keepdim=True)
                  correct += pred.eq(labels.view_as(pred)).sum().item()

        accuracy = (correct / len(testloader.dataset))*100
        print(f'Accuracy of the network on the 10000 test images: {accuracy} %')