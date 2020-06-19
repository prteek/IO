#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/prteek/IO/blob/master/WorldBankDataPlots.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # World Bank data analysis of India and GBR

# In[ ]:


# This cell is not required to be executed (i.e. ignore any error) if Notebook is run locally or in Binder
# Authorise and mount google drive to access code and data files

project_folder = "/content/drive/My Drive/git_repos/IO/"

import os

if os.path.isdir("/content"):
    from google.colab import drive

    drive.mount("/content/drive")

    if not (os.path.isdir(project_folder)):
        os.makedirs(project_folder)
        print("new project folder created")

    os.chdir(project_folder)


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


india_raw_data = pd.read_csv("docs/API_IND_DS2_en_csv_v2_10400058.csv", skiprows=3)
gbr_raw_data = pd.read_csv("docs/API_GBR_DS2_en_csv_v2_10402095.csv", skiprows=3)
india_metadata = pd.read_csv(
    "docs/Metadata_Indicator_API_GBR_DS2_en_csv_v2_10402095.csv", skiprows=0
)
gbr_metadata = pd.read_csv(
    "docs/Metadata_Indicator_API_GBR_DS2_en_csv_v2_10402095.csv", skiprows=0
)


# In[ ]:


years = india_raw_data.columns[5:-1]  # Valid columns for years
india_indicators = india_raw_data["Indicator Name"]
gbr_indicators = gbr_raw_data["Indicator Name"]


# ### User Inputs
# To see all available indicators check the bottom of the notebook

# In[ ]:


keyword = "high-technology exports"

lower_case_indicators = india_indicators.str.lower()
keyword_indicators = india_indicators[
    lower_case_indicators.str.contains(keyword.lower(), regex=False)
]
print("All Indicators matching keyword:\n\n", keyword_indicators, "\n")

desired_index = int(input("Choose the desired index (number in left column above): "))
desired_indicator = india_raw_data["Indicator Name"][desired_index]
print("\n Desired Indicator:\n", desired_indicator)


# In[ ]:


india_yearly_data = [float(india_raw_data[year][desired_index]) for year in years]

gbr_desired_index = gbr_indicators.str.contains(desired_indicator, regex=False)
gbr_yearly_data = [float(gbr_raw_data[year][gbr_desired_index]) for year in years]

years_to_plot = np.array([int(year) for year in years])
bar_width = 0.4

plt.figure()
plt.bar(years_to_plot, india_yearly_data, bar_width, label="India")
plt.bar(years_to_plot + bar_width, gbr_yearly_data, bar_width, label="GBR")
plt.title(desired_indicator)
plt.legend(loc="upper left")
plt.grid()
plt.show()

# Metadata
metadata_index = india_metadata.INDICATOR_NAME.str.contains(
    desired_indicator, regex=False
)
print("Source:\n", list(india_metadata["SOURCE_ORGANIZATION"][metadata_index]), "\n")
print("Note:\n", list(india_metadata["SOURCE_NOTE"][metadata_index]))


# In[ ]:


pd.options.display.max_rows = 2000
#print(india_indicators)
