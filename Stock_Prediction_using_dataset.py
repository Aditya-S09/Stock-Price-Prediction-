#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error


# In[2]:


data = pd.read_csv('Desktop/arc/ADANIPORTS.csv')


# In[32]:


data['MA5'] = data['Close'].rolling(window=5).mean()  # 5-day moving average
data['MA10'] = data['Close'].rolling(window=10).mean()  # 10-day moving average
data['ROC'] = data['Close'].pct_change(periods=10)  #Price change rate over 5 days


# In[33]:


features = ['High', 'Low', 'Close', 'MA5', 'MA10', 'ROC']
target = 'Open' 


# In[34]:


X = data[features].values[10:]  #ignoring first 10 rows for our calculation
y = data[target].values[10:]


# In[35]:


scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[39]:


num_models = 3
models = []
for _ in range(num_models):
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, len(features)), activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    models.append(model)


# In[44]:


for model in models:
    model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)
predictions = []
for model in models:
    predictions.append(model.predict(X_test))


# In[45]:


ensemble_predictions = np.mean(predictions, axis=0)

mse = mean_squared_error(y_test, ensemble_predictions)
print('Mean Squared Error:', mse)


# In[46]:


import matplotlib.pyplot as plt


# In[57]:


plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', s=10)
plt.scatter(range(len(ensemble_predictions)), ensemble_predictions, color='pink', label='Predicted', s=10)
plt.title('Actual vs Predicted Opening Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[53]:


from sklearn.metrics import mean_absolute_error, r2_score


# In[58]:


mae = mean_absolute_error(y_test, ensemble_predictions)
print('Mean Absolute Error (MAE):', mae)

r2 = r2_score(y_test, ensemble_predictions)
print('R-squared (coefficient of determination):', r2)


# In[ ]:




