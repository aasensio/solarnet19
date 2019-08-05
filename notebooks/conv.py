import numpy as np
import torch
import torch.nn as nn

class model_nonlinear(nn.Module):
    def __init__(self):
        super(model_nonlinear, self).__init__()

        self.C1 = nn.Conv1d(1, 10, kernel_size=7, stride=2)
        self.C2 = nn.Conv1d(10, 10, kernel_size=7, stride=2)
        self.linear = nn.Linear(21*10, 2)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        out = self.C1(x)
        out = self.activation(out)
        out = self.C2(out)
        out = self.activation(out)

        out = out.view(x.size(0), -1)

        return self.linear(out)
    
n_profiles = 1000
n_lambda = 100
wvl = np.linspace(-10,10,n_lambda)
v = np.random.uniform(low=-5.0, high=5.0, size=n_profiles)
dv = np.random.uniform(low=0.5, high=1.5, size=n_profiles)
stokes = np.exp(-(wvl[None,:]+v[:,None])**2 / dv[:,None]**2)

stokes = np.expand_dims(stokes, axis=1)

mod2 = model_nonlinear()

optimizer = torch.optim.Adam(mod2.parameters(), lr=5e-2)
loss_fn = nn.MSELoss()

Y = np.array([v, dv]).T
X_torch = torch.from_numpy(stokes.astype('float32'))
Y_torch = torch.from_numpy(Y.astype('float32'))

for loop in range(200):
    optimizer.zero_grad()
    out = mod2(X_torch)
    loss = loss_fn(out, Y_torch)
    loss.backward()
    optimizer.step()
    print(f' It : {loop:3d} - loss : {loss.item():.4f}')