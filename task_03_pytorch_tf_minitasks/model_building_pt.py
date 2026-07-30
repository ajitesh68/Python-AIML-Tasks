import torch 
import torch.nn as nn
import torch.optim as optim 


x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = 2*x+1

class LinearModel(nn.Module):#nn.Module – PyTorch me saare models isko inherit karte hain. Isse parameters(), train(), eval() jaisi functionality milti hai.
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)

model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
#model.parameters() – saare trainable weights (layer ke weight and bias) return karta hai.


for epoch in range(100):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("epoch:", epoch+1, "loss:", loss.item())

print("Learned parameters:", model.linear.weight.item(),model.linear.bias.item())


#Epoch 1: .backward() calculates dw = 2. Optimizer uses dw=2 to update w.
#Agar tune zero nahi kiya, to dw still 2 rahega (memory mein).
#Epoch 2: .backward() calculates dw for new data, maan lo 3. Ab dw = 2 (old) + 3 (new) = 5 ho jayega.
#Zero_grad() ka matlab hai "pichle gradients ko zero kar do, naya start karo".