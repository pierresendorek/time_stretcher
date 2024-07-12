import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def draw_sample():
    if np.random.rand() > 0.3:
        return np.random.randn(1,1) - 3
    
    if np.random.rand() > 0.5:
        return np.random.randn(1,1)
    
    return np.random.randn(1,1) * 0.3


#samples = np.concatenate([draw_sample() for _ in range(10000)])
#plt.hist(samples, bins=100)
#plt.show()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

samples = [draw_sample() for _ in range(10000)]
samples = np.concatenate(samples, axis=0)
samples = torch.tensor(samples, dtype=torch.float32).to(device)


def draw_uniform_on_disc():
    theta = 2 * (np.random.rand(1,1) - 0.5)
    return torch.tensor(theta, dtype=torch.float32).to(device)


# plt.scatter(samples[0,:], samples[1,:])
# plt.show()

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)
        self.activation = nn.LeakyReLU(0.5)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x




model = NN().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for iteration in range(10000):
    optimizer.zero_grad()
    v = draw_uniform_on_disc()
    loss = torch.mean(v * (model(v) - samples) + torch.abs(model(v) - samples)) 
    loss.backward()
    optimizer.step()
    if iteration % 100 == 0:
        print(iteration, loss)


new_samples = []
for _ in range(10000):
    v = draw_uniform_on_disc()
    new_samples.append(model(v))



res = torch.concatenate(new_samples, dim=0)

print(res.shape)

res_np = res.cpu().detach().numpy()

plt.hist(res_np, bins=100, density=True, alpha=0.5)
plt.hist(samples.cpu().detach().numpy(), bins=100, density=True, alpha=0.5)
plt.show()
    
    