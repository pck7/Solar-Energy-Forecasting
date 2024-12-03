#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output1_dataset_daily.csv")
plt.figure(figsize=(100,10))
# Plot ACF
plot_acf(df_w['GHI'], lags=1000)  # Adjust 'lags' as needed
plt.title('Autocorrelation Function (ACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.savefig("acf_daily_r1.png") # save as png
plt.show()


# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output1_weekly_real_dataset.csv")
plt.figure(figsize=(600,10))
# Plot ACF
plot_acf(df_w['GHI'], lags=100)  # Adjust 'lags' as needed
plt.title('Autocorrelation Function (ACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.savefig("acf_weekly_r1.png") # save as png

plt.show()


# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output1_monthly_dataset.csv")
plt.figure(figsize=(600,10))
# Plot ACF
plot_acf(df_w['GHI'], lags=100)  # Adjust 'lags' as needed
plt.title('Autocorrelation Function (ACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.savefig("acf_monthly_r1.png") # save as png

plt.show()


# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output1_dataset_daily.csv")

# Plot PACF
plot_pacf(df_w['GHI'], lags=20)  # Adjust 'lags' as needed
plt.title('Partial Autocorrelation Function (PACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.savefig("pacf_daily_r1.png") # save as png

plt.show()


# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output1_weekly_real_dataset.csv")

# Plot PACF
plot_pacf(df_w['GHI'], lags=15)  # Adjust 'lags' as needed
plt.title('Partial Autocorrelation Function (PACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.savefig("pacf_weekly_r1.png") # save as png

plt.show()


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output1_monthly_dataset.csv")

# Plot PACF
plot_pacf(df_w['GHI'], lags=15)  # Adjust 'lags' as needed
plt.title('Partial Autocorrelation Function (PACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.savefig("pacf_monthly_r1.png") # save as png

plt.show()


# In[ ]:





# In[24]:


# RAJASTHAN 22222


# In[ ]:





# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output2_dataset_daily.csv")
plt.figure(figsize=(100,10))
# Plot ACF
plot_acf(df_w['GHI'], lags=1000)  # Adjust 'lags' as needed
plt.title('Autocorrelation Function (ACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.savefig("acf_daily_r2.png") # save as png

plt.show()


# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output2_weekly_real_dataset.csv")
plt.figure(figsize=(600,10))
# Plot ACF
plot_acf(df_w['GHI'], lags=100)  # Adjust 'lags' as needed
plt.title('Autocorrelation Function (ACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.savefig("acf_weekly_r2.png") # save as png

plt.show()


# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output2_monthly_dataset.csv")
plt.figure(figsize=(600,10))
# Plot ACF
plot_acf(df_w['GHI'], lags=100)  # Adjust 'lags' as needed
plt.title('Autocorrelation Function (ACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.savefig("acf_monthly_r2.png") # save as png

plt.show()


# In[44]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output2_dataset_daily.csv")

# Plot PACF
plot_pacf(df_w['GHI'], lags=20)  # Adjust 'lags' as needed
plt.title('Partial Autocorrelation Function (PACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.savefig("pacf_daily_r2.png") # save as png

plt.show()


# In[45]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output2_weekly_real_dataset.csv")

# Plot PACF
plot_pacf(df_w['GHI'], lags=15)  # Adjust 'lags' as needed
plt.title('Partial Autocorrelation Function (PACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.savefig("pacf_weekly_r2.png") # save as png

plt.show()


# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# Assuming you have a weekly dataset in a pandas DataFrame with a column named 'value'
# You can replace this with your actual dataset and column name
# For example, if your DataFrame is named df and the column is 'weekly_data', use df['weekly_data']
df_w = pd.read_csv("output2_monthly_dataset.csv")

# Plot PACF
plot_pacf(df_w['GHI'], lags=15)  # Adjust 'lags' as needed
plt.title('Partial Autocorrelation Function (PACF) Plot')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.savefig("pacf_monthly_r2.png") # save as png

plt.show()


# In[ ]:




