# ========================================================================
# Alternative Hypothesis: The players previous seasons performance and the 
#average age regression has an affect on their next season.
# Null Hypothesis: The players previous seasons performance and the average 
#age regression doesnt not have an affect on their next season.

# Ommiting 2020, because of Covid year and only 60 game season

# ========================================================================
# Nice user interface with two options


# Goal 1: Predict that players next seasons performance with all features
#   inputs:
#       Name
#   Output:
#       All features


# Goal 2: Import certain made up stats and predict your players next season
#   Input:
#       Age, Seasons Played, WAR, ~3 others stats
#   Output:
#       Inputs but modified for next season

# ========================================================================
# Step 1: import data and libraries
# Step 2: Data Exploration
# Step 3: Train/Test Split (Split by year)
# Step 4: Scale/Normalize data
# Step 5: Train Models
# Step 6: Evaluate Models (Visualizations and Metrics)
# Step 7: User Interface (Streamlit)
# ========================================================================

# ========================================================================
# import data and libraries (Step 1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

PD = pd.read_csv('pitchers_war_dataset_enhanced.csv')
HD = pd.read_csv('hitters_war_dataset_enhanced.csv')
# ========================================================================


# ========================================================================
# Data Exploration (Step 2)

# WAR by Season Visualizations
# plt.bar(PD['Season'], PD['WAR'])
# plt.title('Pitcher WAR by Season')
# plt.savefig('Pitcher_WAR_by_Season.png')
# plt.show()

# plt.bar(HD['Season'], HD['WAR'])
# plt.title('Hitter WAR by Season')
# plt.savefig('Hitter_WAR_by_Season.png')
# plt.show()



# Average Feature Values Visualization
# features = ['AVG', 'RBI', 'K%']

# Average_AVG = HD['AVG'].mean()
# Average_RBI = HD['RBI'].mean()
# Average_KP = PD['K%'].mean()

# plt.bar(features, [Average_AVG, Average_RBI, Average_KP])
# plt.ylabel('Value')
# plt.title('Average Feature Values')
# plt.savefig('Average_Feature_Values_not_normalized.png')
# plt.show()
# ==========================================================================

# ==========================================================================
# Train/Test Split (Split by year) (Step 3)


