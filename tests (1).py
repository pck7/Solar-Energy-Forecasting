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





# In[ ]:





# In[ ]:





# In[ ]:





# ADF TEST

# In[1]:


import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load your time series data into a pandas DataFrame
# Replace 'your_time_series.csv' with the actual file path or URL
df = pd.read_csv('output1_weekly_real_dataset.csv')

# Extract the time series values (assuming a column named 'values')
time_series = df['GHI']

# Perform Augmented Dickey-Fuller test
result = adfuller(time_series)

# Extract and print the test statistics and p-value
adf_statistic = result[0]
p_value = result[1]

print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')

# Compare the p-value to a significance level (e.g., 0.05) to decide whether to reject the null hypothesis
if p_value <= 0.01:
    print("Reject the null hypothesis. The time series is likely stationary.")
else:
    print("Fail to reject the null hypothesis. The time series may be non-stationary.")


# In[2]:


import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load your time series data into a pandas DataFrame
# Replace 'your_time_series.csv' with the actual file path or URL
df = pd.read_csv('output1_dataset_daily.csv')

# Extract the time series values (assuming a column named 'values')
time_series = df['GHI']

# Perform Augmented Dickey-Fuller test
result = adfuller(time_series)

# Extract and print the test statistics and p-value
adf_statistic = result[0]
p_value = result[1]

print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')

# Compare the p-value to a significance level (e.g., 0.05) to decide whether to reject the null hypothesis
if p_value <= 0.01:
    print("Reject the null hypothesis. The time series is likely stationary.")
else:
    print("Fail to reject the null hypothesis. The time series may be non-stationary.")


# In[3]:


import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load your time series data into a pandas DataFrame
# Replace 'your_time_series.csv' with the actual file path or URL
df = pd.read_csv('output2_weekly_real_dataset.csv')

# Extract the time series values (assuming a column named 'values')
time_series = df['GHI']

# Perform Augmented Dickey-Fuller test
result = adfuller(time_series)

# Extract and print the test statistics and p-value
adf_statistic = result[0]
p_value = result[1]

print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')

# Compare the p-value to a significance level (e.g., 0.05) to decide whether to reject the null hypothesis
if p_value <= 0.01:
    print("Reject the null hypothesis. The time series is likely stationary.")
else:
    print("Fail to reject the null hypothesis. The time series may be non-stationary.")


# In[4]:


import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load your time series data into a pandas DataFrame
# Replace 'your_time_series.csv' with the actual file path or URL
df = pd.read_csv('output2_dataset_daily.csv')

# Extract the time series values (assuming a column named 'values')
time_series = df['GHI']

# Perform Augmented Dickey-Fuller test
result = adfuller(time_series)

# Extract and print the test statistics and p-value
adf_statistic = result[0]
p_value = result[1]

print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')

# Compare the p-value to a significance level (e.g., 0.05) to decide whether to reject the null hypothesis
if p_value <= 0.01:
    print("Reject the null hypothesis. The time series is likely stationary.")
else:
    print("Fail to reject the null hypothesis. The time series may be non-stationary.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


import pandas as pd
from statsmodels.tsa.stattools import kpss

# Load your time series data into a pandas DataFrame
# Replace 'your_time_series.csv' with the actual file path or URL
df = pd.read_csv('output1_weekly_real_dataset.csv')

# Extract the time series values (assuming a column named 'values')
time_series = df['GHI']

# Perform KPSS test
result = kpss(time_series, regression='c')  # 'c' for a constant term, 'ct' for a constant and trend

# Extract and print the test statistics and p-value
kpss_statistic = result[0]
p_value = result[1]

print(f'KPSS Statistic: {kpss_statistic}')
print(f'p-value: {p_value}')

# Compare the p-value to a significance level (e.g., 0.05) to decide whether to reject the null hypothesis
if p_value <= 0.01:
    print("Reject the null hypothesis. The time series is likely non-stationary.")
else:
    print("Fail to reject the null hypothesis. The time series is likely stationary.")


# In[6]:


import pandas as pd
from statsmodels.tsa.stattools import kpss

# Load your time series data into a pandas DataFrame
# Replace 'your_time_series.csv' with the actual file path or URL
df = pd.read_csv('output1_dataset_daily.csv')

# Extract the time series values (assuming a column named 'values')
time_series = df['GHI']

# Perform KPSS test
result = kpss(time_series, regression='c')  # 'c' for a constant term, 'ct' for a constant and trend

# Extract and print the test statistics and p-value
kpss_statistic = result[0]
p_value = result[1]

print(f'KPSS Statistic: {kpss_statistic}')
print(f'p-value: {p_value}')

# Compare the p-value to a significance level (e.g., 0.05) to decide whether to reject the null hypothesis
if p_value <= 0.01:
    print("Reject the null hypothesis. The time series is likely non-stationary.")
else:
    print("Fail to reject the null hypothesis. The time series is likely stationary.")


# In[7]:


import pandas as pd
from statsmodels.tsa.stattools import kpss

# Load your time series data into a pandas DataFrame
# Replace 'your_time_series.csv' with the actual file path or URL
df = pd.read_csv('output2_weekly_real_dataset.csv')

# Extract the time series values (assuming a column named 'values')
time_series = df['GHI']

# Perform KPSS test
result = kpss(time_series, regression='c')  # 'c' for a constant term, 'ct' for a constant and trend

# Extract and print the test statistics and p-value
kpss_statistic = result[0]
p_value = result[1]

print(f'KPSS Statistic: {kpss_statistic}')
print(f'p-value: {p_value}')

# Compare the p-value to a significance level (e.g., 0.05) to decide whether to reject the null hypothesis
if p_value <= 0.01:
    print("Reject the null hypothesis. The time series is likely non-stationary.")
else:
    print("Fail to reject the null hypothesis. The time series is likely stationary.")


# In[8]:


import pandas as pd
from statsmodels.tsa.stattools import kpss

# Load your time series data into a pandas DataFrame
# Replace 'your_time_series.csv' with the actual file path or URL
df = pd.read_csv('output2_dataset_daily.csv')

# Extract the time series values (assuming a column named 'values')
time_series = df['GHI']

# Perform KPSS test
result = kpss(time_series, regression='c')  # 'c' for a constant term, 'ct' for a constant and trend

# Extract and print the test statistics and p-value
kpss_statistic = result[0]
p_value = result[1]

print(f'KPSS Statistic: {kpss_statistic}')
print(f'p-value: {p_value}')

# Compare the p-value to a significance level (e.g., 0.05) to decide whether to reject the null hypothesis
if p_value <= 0.01:
    print("Reject the null hypothesis. The time series is likely non-stationary.")
else:
    print("Fail to reject the null hypothesis. The time series is likely stationary.")


# In[ ]:




