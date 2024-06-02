# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable

# Load the dataset
df = pd.read_csv('coin_Aave.csv')

# Ensure data is loaded correctly
print(df.head())

# Plot the original data
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close Price History')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Bitcoin Close Price History')
plt.legend()
plt.show()

# Data preprocessing
data = df.filter(['Close']).values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define a function to create the training dataset
def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Define the time step for the LSTM model
time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape data for LSTM model
X = torch.tensor(X.reshape(X.shape[0], X.shape[1], 1), dtype=torch.float32)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# Splitting the data into training and testing sets
train_size = int(len(scaled_data) * 0.7)
test_size = len(scaled_data) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model
class LSTMModel(nn.Module):
    def _init_(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self)._init_()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1

# Instantiate the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Transform data back to original scale
train_predict = model(X_train).detach().numpy()
test_predict = test_outputs.detach().numpy()
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Prepare the data for plotting
train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict)+time_step, :] = train_predict

test_plot = np.empty_like(data)
test_plot[:, :] = np.nan
test_plot[len(train_predict)+(time_step*2)+1:len(data)-1, :] = test_predict

# Plot the predictions
plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(train_plot, label='Training Predictions')
plt.plot(test_plot, label='Testing Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Bitcoin Price Prediction using LSTM')
plt.legend()
plt.show()
