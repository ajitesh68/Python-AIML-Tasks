from inspect import Parameter
from tensorflow.random import shuffle
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset , Dataloader


torch.manual.seed(42)
X = torch.randn(100,1) * 10 
y = 2 * X +1 + torch.randn(100,1) * 0.5

dataset = TensorDataset(X,y)
dataloader = Dataloader(dataset,batch_size=8, shuffle = True)


model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 50 
for epoch in range(epochs):
    for batch_x,batch_y in dataloader:
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    
    if epoch%10==0:
        with torch.no_grad():
            total_loss = criterion(model(X), y)
            print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
