#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

df_w = pd.read_csv("output1_dataset_daily.csv")

# Assume df contains the preprocessed daily/weekly data.
# df['GHI'] has the GHI values
X = df_w['GHI'].values
ans = seasonal_decompose(X, model='additive', period=365)

# Set up subplots
fig, axes = plt.subplots(4, 1, figsize=(30, 10), sharex=True)

# Plot the decomposed components separately
axes[0].plot(ans.observed, label='Observed')
axes[0].legend()
axes[0].set_title('Observed Component')

axes[1].plot(ans.trend, label='Trend')
axes[1].legend()
axes[1].set_title('Trend Component')

axes[2].plot(ans.seasonal, label='Seasonal')
axes[2].legend()
axes[2].set_title('Seasonal Component')

axes[3].plot(ans.resid, label='Residual')
axes[3].legend()
axes[3].set_title('Residual Component')
plt.savefig("daily_rajasthan1.png") # save as png
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[10]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

df_w = pd.read_csv("output1_weekly_real_dataset.csv")

# Assume df contains the preprocessed daily/weekly data.
# df['GHI'] has the GHI values
X = df_w['GHI'].values
ans = seasonal_decompose(X, model='additive', period=52)

# Set up subplots
fig, axes = plt.subplots(4, 1, figsize=(30, 10), sharex=True)

# Plot the decomposed components separately
axes[0].plot(ans.observed, label='Observed')
axes[0].legend()
axes[0].set_title('Observed Component')

axes[1].plot(ans.trend, label='Trend')
axes[1].legend()
axes[1].set_title('Trend Component')

axes[2].plot(ans.seasonal, label='Seasonal')
axes[2].legend()
axes[2].set_title('Seasonal Component')

axes[3].plot(ans.resid, label='Residual')
axes[3].legend()
axes[3].set_title('Residual Component')

# Adjust layout
plt.tight_layout()
plt.savefig("weekly_rajasthan1.png") # save as png

# Show the plots
plt.show()


# In[8]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

df_w = pd.read_csv("output1_monthly_dataset.csv")

# Assume df contains the preprocessed daily/weekly data.
# df['GHI'] has the GHI values
X = df_w['GHI'].values
ans = seasonal_decompose(X, model='additive', period=12)

# Set up subplots
fig, axes = plt.subplots(4, 1, figsize=(30, 10), sharex=True)

# Plot the decomposed components separately
axes[0].plot(ans.observed, label='Observed')
axes[0].legend()
axes[0].set_title('Observed Component')

axes[1].plot(ans.trend, label='Trend')
axes[1].legend()
axes[1].set_title('Trend Component')

axes[2].plot(ans.seasonal, label='Seasonal')
axes[2].legend()
axes[2].set_title('Seasonal Component')

axes[3].plot(ans.resid, label='Residual')
axes[3].legend()
axes[3].set_title('Residual Component')

# Adjust layout
plt.tight_layout()
plt.savefig("monthly_rajasthan1.png") # save as png

# Show the plots
plt.show()


# In[7]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

df_w = pd.read_csv("output2_dataset_daily.csv")

# Assume df contains the preprocessed daily/weekly data.
# df['GHI'] has the GHI values
X = df_w['GHI'].values
ans = seasonal_decompose(X, model='additive', period=365)

# Set up subplots
fig, axes = plt.subplots(4, 1, figsize=(30, 10), sharex=True)

# Plot the decomposed components separately
axes[0].plot(ans.observed, label='Observed')
axes[0].legend()
axes[0].set_title('Observed Component')

axes[1].plot(ans.trend, label='Trend')
axes[1].legend()
axes[1].set_title('Trend Component')

axes[2].plot(ans.seasonal, label='Seasonal')
axes[2].legend()
axes[2].set_title('Seasonal Component')

axes[3].plot(ans.resid, label='Residual')
axes[3].legend()
axes[3].set_title('Residual Component')

# Adjust layout
plt.tight_layout()
plt.savefig("daily_rajasthan2.png") # save as png

# Show the plots
plt.show()


# In[6]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

df_w = pd.read_csv("output2_weekly_real_dataset.csv")

# Assume df contains the preprocessed daily/weekly data.
# df['GHI'] has the GHI values
X = df_w['GHI'].values
ans = seasonal_decompose(X, model='additive', period=52)

# Set up subplots
fig, axes = plt.subplots(4, 1, figsize=(30, 10), sharex=True)

# Plot the decomposed components separately
axes[0].plot(ans.observed, label='Observed')
axes[0].legend()
axes[0].set_title('Observed Component')

axes[1].plot(ans.trend, label='Trend')
axes[1].legend()
axes[1].set_title('Trend Component')

axes[2].plot(ans.seasonal, label='Seasonal')
axes[2].legend()
axes[2].set_title('Seasonal Component')

axes[3].plot(ans.resid, label='Residual')
axes[3].legend()
axes[3].set_title('Residual Component')

# Adjust layout
plt.tight_layout()
plt.savefig("weekly_rajasthan2.png") # save as png

# Show the plots
plt.show()


# In[5]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

df_w = pd.read_csv("output2_monthly_dataset.csv")

# Assume df contains the preprocessed daily/weekly data.
# df['GHI'] has the GHI values
X = df_w['GHI'].values
ans = seasonal_decompose(X, model='additive', period=12)

# Set up subplots
fig, axes = plt.subplots(4, 1, figsize=(30, 10), sharex=True)

# Plot the decomposed components separately
axes[0].plot(ans.observed, label='Observed')
axes[0].legend()
axes[0].set_title('Observed Component')

axes[1].plot(ans.trend, label='Trend')
axes[1].legend()
axes[1].set_title('Trend Component')

axes[2].plot(ans.seasonal, label='Seasonal')
axes[2].legend()
axes[2].set_title('Seasonal Component')

axes[3].plot(ans.resid, label='Residual')
axes[3].legend()
axes[3].set_title('Residual Component')

# Adjust layout
plt.tight_layout()
plt.savefig("monthly_rajasthan2.png") # save as png
# Show the plots
plt.show()


# In[ ]:




