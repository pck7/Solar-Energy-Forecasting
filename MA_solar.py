#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd


# In[31]:


df = pd.read_csv('output1_weekly_real_dataset.csv')


# In[32]:


df


# In[34]:


import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from math import sqrt

# Ensure the 'week' column is set as the index
# df.set_index('Week', inplace=True)

# Hyperparameter tuning
best_mae = float('inf')
best_q = None

for q in range(1,20):  # Experiment with different values of q
    model = ARIMA(df['GHI'], order=(0, 0, q), trend='c')  # Adjust order based on your model
    model_fit = model.fit()

    # Predict on the entire dataset for visualization
    predictions = model_fit.predict(start=df['Week'].iloc[0], end=df['Week'].iloc[-1], dynamic=False)

    # Evaluate the model
    mae = mean_absolute_error(df['GHI'], predictions)

    # Print and update best hyperparameter
    print(f'MA({q}) - MAE: {mae:.2f}')
    if mae < best_mae:
        best_mae = mae
        best_q = q

# Train the best model
final_model = ARIMA(df['GHI'], order=(0, 0, best_q), trend='c')
final_model_fit = final_model.fit()

# Predict on the entire dataset for visualization
all_predictions = final_model_fit.predict(start=df['Week'].iloc[0], end=df['Week'].iloc[-1] + 10, dynamic=False)

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(df['Week'], df['GHI'], label='Actual')
plt.plot(all_predictions.index, all_predictions, label=f'Predicted (MA({best_q}))',color='orange')
plt.title('Moving Average Model: Actual vs Predicted GHI Values')
plt.xlabel('Week')
plt.ylabel('GHI')
start_point = 620
end_point = max(df['Week'].max(), all_predictions.index.max())
plt.xlim(start_point, end_point)
# Ensure ticks are aligned with the desired increments (every 20 weeks)
tick_positions = np.arange(start_point, end_point, 20)
plt.xticks(tick_positions) 
plt.legend()
plt.show()

# Print the best hyperparameter 'q'
print(f'Best hyperparameter (q): {best_q}')

# Calculate MAPE
mape = np.mean(np.abs((df['GHI'] - all_predictions[:len(df)]) / df['GHI'])) * 100
print(f'MAPE: {mape:.2f}%')

# Print the MAE for the best model
print(f'MAE for the best model: {best_mae:.2f}')


# In[38]:


df1 = pd.read_csv('output1_dataset_daily.csv')


# In[39]:


df1


# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from math import sqrt

df1['new_column'] = range(5475)

# Hyperparameter tuning
best_mae = float('inf')
best_q = None

for q in range(1, 38):  # Experiment with different values of q
    model = ARIMA(df1['GHI'], order=(0, 0, q), trend='c')  # Adjust order based on your model
    model_fit = model.fit()

    # Predict on the entire dataset for visualization
    predictions = model_fit.predict(start=df1['new_column'].iloc[0], end=df1['new_column'].iloc[-1], dynamic=False)

    # Evaluate the model
    mae = mean_absolute_error(df1['GHI'], predictions)

    # Print and update best hyperparameter
    print(f'MA({q}) - MAE: {mae:.2f}')
    if mae < best_mae:
        best_mae = mae
        best_q = q

# Train the best model
final_model = ARIMA(df1['GHI'], order=(0, 0, best_q), trend='c')
final_model_fit = final_model.fit()

# Predict on the entire dataset for visualization
all_predictions = final_model_fit.predict(start=df1['new_column'].iloc[0], end=df1['new_column'].iloc[-1] + 10, dynamic=False)

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(df1['new_column'], df1['GHI'], label='Actual')
plt.plot(all_predictions.index, all_predictions, label=f'Predicted (MA({best_q}))',color='orange')
plt.title('Moving Average Model: Actual vs Predicted GHI Values')
plt.xlabel('Day Code')
plt.ylabel('GHI')
plt.legend()
plt.show()

# Print the best hyperparameter 'q'
print(f'Best hyperparameter (q): {best_q}')

# Calculate MAPE
mape = np.mean(np.abs((df1['GHI'] - all_predictions[:len(df1)]) / df1['GHI'])) * 100
print(f'MAPE: {mape:.2f}%')

# Print the MAE for the best model
print(f'MAE for the best model: {best_mae:.2f}')


# In[47]:


plt.figure(figsize=(12, 6))
plt.plot(df1['new_column'], df1['GHI'], label='Actual')
plt.plot(all_predictions.index, all_predictions, label=f'Predicted (MA({best_q}))',color='orange')
plt.xlim(4400, max(df['Week'].max(), all_predictions.index.max() + 10))
plt.title('Moving Average Model: Actual vs Predicted GHI Values')
plt.xlabel('Day Code')
plt.ylabel('GHI')
plt.legend()
plt.show()


# In[ ]:




