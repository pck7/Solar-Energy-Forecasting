#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from scipy import stats

# Load your weekly dataset file into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path or URL
df = pd.read_csv('output1_weekly_real_dataset.csv')

# Extract the values from the dataset (assuming a column named 'values')
data = df['GHI']

# List of distributions to test
distributions = ['weibull_min', 'weibull_max', 'norm', 'gamma', 'expon', 'lognorm', 'beta']

# Perform KS test for each distribution
for distribution in distributions:
    print(f"{distribution.capitalize()} (p-value):")
    
    # Fit the distribution to the data
    params = getattr(stats, distribution).fit(data)
    
    # Perform KS test
    ks_statistic, p_value = stats.kstest(data, distribution, args=params)
    
    # Print the p-value
    print(f"{p_value:.4e}\n")


# In[4]:


import pandas as pd
from scipy import stats

# Load your weekly dataset file into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path or URL
df = pd.read_csv('output1_dataset_daily.csv')

# Extract the values from the dataset (assuming a column named 'values')
data = df['GHI']

# List of distributions to test
distributions = ['weibull_min', 'weibull_max', 'norm', 'gamma', 'expon', 'lognorm', 'beta']

# Perform KS test for each distribution
for distribution in distributions:
    print(f"{distribution.capitalize()} (p-value):")
    
    # Fit the distribution to the data
    params = getattr(stats, distribution).fit(data)
    
    # Perform KS test
    ks_statistic, p_value = stats.kstest(data, distribution, args=params)
    
    # Print the p-value
    print(f"{p_value:.4e}\n")


# In[5]:


import pandas as pd
from scipy import stats

# Load your weekly dataset file into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path or URL
df = pd.read_csv('output1_monthly_dataset.csv')

# Extract the values from the dataset (assuming a column named 'values')
data = df['GHI']

# List of distributions to test
distributions = ['weibull_min', 'weibull_max', 'norm', 'gamma', 'expon', 'lognorm', 'beta']

# Perform KS test for each distribution
for distribution in distributions:
    print(f"{distribution.capitalize()} (p-value):")
    
    # Fit the distribution to the data
    params = getattr(stats, distribution).fit(data)
    
    # Perform KS test
    ks_statistic, p_value = stats.kstest(data, distribution, args=params)
    
    # Print the p-value
    print(f"{p_value:.4e}\n")


# In[ ]:





# In[ ]:





# In[6]:


#  RAJASTHAN 22222


# In[ ]:





# In[ ]:





# In[7]:


import pandas as pd
from scipy import stats

# Load your weekly dataset file into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path or URL
df = pd.read_csv('output2_weekly_real_dataset.csv')

# Extract the values from the dataset (assuming a column named 'values')
data = df['GHI']

# List of distributions to test
distributions = ['weibull_min', 'weibull_max', 'norm', 'gamma', 'expon', 'lognorm', 'beta']

# Perform KS test for each distribution
for distribution in distributions:
    print(f"{distribution.capitalize()} (p-value):")
    
    # Fit the distribution to the data
    params = getattr(stats, distribution).fit(data)
    
    # Perform KS test
    ks_statistic, p_value = stats.kstest(data, distribution, args=params)
    
    # Print the p-value
    print(f"{p_value:.4e}\n")


# In[9]:


import pandas as pd
from scipy import stats

# Load your weekly dataset file into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path or URL
df = pd.read_csv('output2_dataset_daily.csv')

# Extract the values from the dataset (assuming a column named 'values')
data = df['GHI']

# List of distributions to test
distributions = ['weibull_min', 'weibull_max', 'norm', 'gamma', 'expon', 'lognorm', 'beta']

# Perform KS test for each distribution
for distribution in distributions:
    print(f"{distribution.capitalize()} (p-value):")
    
    # Fit the distribution to the data
    params = getattr(stats, distribution).fit(data)
    
    # Perform KS test
    ks_statistic, p_value = stats.kstest(data, distribution, args=params)
    
    # Print the p-value
    print(f"{p_value:.4e}\n")


# In[10]:


import pandas as pd
from scipy import stats

# Load your weekly dataset file into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path or URL
df = pd.read_csv('output2_monthly_dataset.csv')

# Extract the values from the dataset (assuming a column named 'values')
data = df['GHI']

# List of distributions to test
distributions = ['weibull_min', 'weibull_max', 'norm', 'gamma', 'expon', 'lognorm', 'beta']

# Perform KS test for each distribution
for distribution in distributions:
    print(f"{distribution.capitalize()} (p-value):")
    
    # Fit the distribution to the data
    params = getattr(stats, distribution).fit(data)
    
    # Perform KS test
    ks_statistic, p_value = stats.kstest(data, distribution, args=params)
    
    # Print the p-value
    print(f"{p_value:.4e}\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
import numpy as np

# Load your dataset file into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path or URL
df = pd.read_csv('output1_weekly_real_dataset.csv')

# Extract the values from the dataset (assuming a column named 'values')
data = df['GHI']

# Fit the beta distribution to the data
params = stats.beta.fit(data)

# Generate data points for the fitted distribution
xmin, xmax = min(data), max(data)
x = np.linspace(xmin, xmax, 1000)
pdf_fitted = stats.beta.pdf(x, *params)

# Plot histogram of the data
plt.hist(data, density=True, alpha=0.6, color='g', bins=30, label='Histogram')

# Plot the PDF of the fitted beta distribution
plt.plot(x, pdf_fitted, 'r-', lw=2, label='Beta PDF')

# Add labels and a legend
plt.title('Fitted Beta Distribution')
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig("beta_weekly_rajathan1.png") # save as png

# Show the plot
plt.show()


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
import numpy as np

# Load your dataset file into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path or URL
df = pd.read_csv('output2_weekly_real_dataset.csv')

# Extract the values from the dataset (assuming a column named 'values')
data = df['GHI']

# Fit the beta distribution to the data
params = stats.beta.fit(data)

# Generate data points for the fitted distribution
xmin, xmax = min(data), max(data)
x = np.linspace(xmin, xmax, 1000)
pdf_fitted = stats.beta.pdf(x, *params)

# Plot histogram of the data
plt.hist(data, density=True, alpha=0.6, color='g', bins=30, label='Histogram')

# Plot the PDF of the fitted beta distribution
plt.plot(x, pdf_fitted, 'r-', lw=2, label='Beta PDF')

# Add labels and a legend
plt.title('Fitted Beta Distribution')
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig("beta_weekly_rajathan2.png") # save as png

# Show the plot
plt.show()


# In[ ]:




