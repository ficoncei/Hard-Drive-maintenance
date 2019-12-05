# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

samples=21600

device = torch.device("cuda:0")

#MLP definition

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,4)
        self.fc2 = nn.Linear(4,4)
        self.fc3 = nn.Linear(4,1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

model=  Net()

#Function that loads data and uses it to train the network
def trainNN():    
    print("Training ongoing...")
    HWdata = pd.read_csv('trainingdataset.csv')
    HWdatanp=pd.DataFrame(HWdata).to_numpy()

    for sample in range(samples):
        inputLayer = torch.tensor([HWdatanp[sample]]).float()
        faultProbability = model(inputLayer)

        loss_function = torch.nn.MSELoss()
        loss=loss_function(HWdatanp[sample,0],faultProbability)  
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
#Function that receives as user input an HD ID and outputs a failure probability
def testNN(HD_ID,threshold):
    HWdata2 = pd.read_csv('testdataset.csv')
    HWdatanp2=pd.DataFrame(HWdata2).to_numpy()
    
    inputLayer = torch.tensor([HWdatanp2[HD_ID]]).float()
    faultProbability = model(inputLayer)
    print(faultProbability)
    if faultProbability>threshold:
        print("REPLACE HD...")

#Main
trainNN()

#After running the script for the first time, test different 
#values on the console with the command:
testNN(4,0.3)


