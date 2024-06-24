#!/usr/bin/env python
# coding: utf-8

# In this notebook we will be building and training LSTM to predict IBM stock. We will use PyTorch.

# In[22]:


import torch
#device = torch_directml.device()


# ## 1. Libraries and settings

# In[23]:


import numpy as np
import random
import pandas as pd 
from pylab import mpl, plt
#plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('env', 'HSA_OVERRIDE_GFX_VERSION=11.0.0')
#from pandas import datetime
import math, time

import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable

#device = torch_directml.device()
device = torch.device("cuda")
#device = torch.device("cpu")


import os
for dirname, _, filenames in os.walk('./input'):
    for i, filename in enumerate(filenames):
        if i<5:
            print(os.path.join(dirname,filename))


# In[ ]:





# ## 2. Load data

# In[24]:


pd.read_csv("./input/Data/Stocks/goog.us.txt")


# In[25]:


def stocks_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        df_temp = pd.read_csv("./input/Data/Stocks/{}.us.txt".format(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Close': symbol})
        df = df.join(df_temp)
    return df


# In[26]:


dates = pd.date_range('2015-01-02','2016-12-31',freq='B')
symbols = ['goog','ibm','aapl']
df = stocks_data(symbols, dates)
df.fillna(method='pad')
df.plot(figsize=(10, 6), subplots=True);


# In[27]:


df.head()


# In[28]:


dates = pd.date_range('2010-01-02','2017-10-11',freq='B')
df1=pd.DataFrame(index=dates)
df_ibm=pd.read_csv("./input/Data/Stocks/ibm.us.txt", parse_dates=True, index_col=0)
df_ibm=df1.join(df_ibm)
df_ibm[['Close']].plot(figsize=(15, 6))
plt.ylabel("stock_price")
plt.title("IBM Stock")
plt.show()


# In[29]:


df_ibm=df_ibm[['Close']]
df_ibm.info()


# In[30]:


df_ibm=df_ibm.fillna(method='ffill')

scaler = MinMaxScaler(feature_range=(-1, 1))
df_ibm['Close'] = scaler.fit_transform(df_ibm['Close'].values.reshape(-1,1))
#df_ibm


# In[31]:


# function to create train, test data given stock data and sequence length
def load_data(stock, look_back):
    data_raw = stock.values # convert to numpy array
    data = []
    
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

look_back = 60 # choose sequence length
x_train, y_train, x_test, y_test = load_data(df_ibm, look_back)
#x_train.to(device)
#y_train.to(device)
#x_test.to(device)
#y_test.to(device)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)


# In[32]:


# make training and test sets in torch
x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)


# In[33]:


y_train.size(),x_train.size()


# ## 3. Build the structure of model

# In[34]:


# Build model
#####################
input_dim = 1
hidden_dim = 256
num_layers = 4
output_dim = 1


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.to(device)


        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = c0.to(device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        hn = hn.to(device)
        cn = cn.to(device)
        oot = out.to(device)

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :])

        # out.size() --> 100, 10
        return out
    

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

model.to(device)

loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())


# In[ ]:





# In[35]:


get_ipython().run_cell_magic('time', '', '# Train model\n#####################\nnum_epochs = 100\nhist = np.zeros(num_epochs)\n\n# Number of steps to unroll\nseq_dim =look_back-1  \n#x_train = x_train.to("cuda")\n#model.to("cuda")\nfor t in range(num_epochs):\n    # Initialise hidden state\n    # Don\'t do this if you want your LSTM to be stateful\n    #model.hidden = model.init_hidden()\n    #print(1)\n    #print(x_train.device)\n    # Forward pass\n\n\n    #print(next(model.parameters()).device)\n\n    y_train_pred = model(x_train)\n\n    loss = loss_fn(y_train_pred, y_train)\n    if t % 10 == 0 and t !=0:\n        print("Epoch ", t, "MSE: ", loss.item())\n    hist[t] = loss.item()\n\n    # Zero out gradient, else they will accumulate between epochs\n    optimiser.zero_grad()\n\n    # Backward pass\n    loss.backward()\n\n    # Update parameters\n    optimiser.step()\n')


# In[ ]:





# In[36]:


plt.plot(hist, label="Training loss")
plt.legend()
plt.show()


# In[ ]:





# In[37]:


np.shape(y_train_pred)


# In[38]:


# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.to("cpu").detach().numpy())
y_train = scaler.inverse_transform(y_train.to("cpu").detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.to("cpu").detach().numpy())
y_test = scaler.inverse_transform(y_test.to("cpu").detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[39]:


# Visualising the results
figure, axes = plt.subplots(figsize=(15, 6))
axes.xaxis_date()

axes.plot(df_ibm[len(df_ibm)-len(y_test):].index, y_test, color = 'red', label = 'Real IBM Stock Price')
axes.plot(df_ibm[len(df_ibm)-len(y_test):].index, y_test_pred, color = 'blue', label = 'Predicted IBM Stock Price')
#axes.xticks(np.arange(0,394,50))
plt.title('IBM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('IBM Stock Price')
plt.legend()
plt.savefig('ibm_pred.png')
plt.show()


# In[40]:


y_test_pred


# In[41]:


model


# In[42]:


torch.save(model,"model123")


# In[ ]:




