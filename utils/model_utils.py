from tqdm import trange
import torch
import torch.nn as nn

class PyTorchCNN(nn.Module):
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 28 x 28 x 1 -> 28 x 28 x 16
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28 x 28 x 16 -> 14 x 14 x 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 14 x 14 x 16 -> 14 x 14 x 32
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14 x 14 x 32 -> 7 x 7 x 32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 7 x 7 x 32 -> 7 x 7 x 64
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 3) # (7 x 7 x 64) -> 3
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x)) 
        x = self.pool2(x)
        x = self.relu3(self.conv3(x))
        x = self.flatten(x.permute(0, 2, 3, 1)) # (b x 64 x 7 x 7) -> (b x 7 x 7 x 64) -> (3136)
        x = self.relu4(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def train(model, X, y, lr=0.0001, num_epochs=2, batch_size=256):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        with trange(0, X.shape[0], batch_size, desc=f'Training (Epoch {epoch+1})') as progress_bar:
            for b in progress_bar:
                optimizer.zero_grad()
                preds = model(X[b:b+batch_size])
                loss_fn = nn.BCELoss()
                loss = loss_fn(preds, y[b:b+batch_size].view(-1, 1))
                progress_bar.set_postfix({'loss': loss.item()})
                loss.backward()
                optimizer.step()

def predict(model, X, batch_size=256):
    model.eval()
    y_pred = None
    for b in trange(0, X.shape[0], batch_size, desc='Evaluating'):
        with torch.no_grad():
            y_pred_batch = model(X[b:b+batch_size]).round().flatten()
        if y_pred is None:
            y_pred = y_pred_batch
        else:
            y_pred = torch.cat((y_pred, y_pred_batch))
    return y_pred

def evaluate(model, X, y, batch_size=256):
    y_pred = predict(model, X, batch_size)
    accuracy = (y_pred == y).float().mean()
    return accuracy.item()

