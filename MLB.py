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
#       Hitters:
#       WAR, H, HR, RBI, BB%, K%, AVG, OBP, SLG, wOBA, wRC+
#       Pitchers:
#       WAR, W, L, IP, ER, SO, K%, BB%, K-BB%, HR/9, WHIP, FIP, ERA

# Goal 2: Import certain made up stats and predict your players next season
#   Input:
#       Age, Seasons Played, WAR, ~3 others stats
#   Output:
#       WAR_next 

# ========================================================================
# Step 1: Import data and libraries
# Step 2: Data Exploration
# Step 3: Scale/Normalize & Split data
# Step 4: Train Models
# Step 5: Evaluate Models (Visualizations and Metrics)
# Step 6: User Interface (Streamlit)
# ========================================================================

# ========================================================================
# Import data and libraries (Step 1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
import sklearn.metrics as metrics
import xgboost as xgb
import tensorflow as tf
import pickle





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
# Train/Test Split (Split by year) and Scale/Normalize data (Step 3)

# Split players that fall under years to make it 80 Train and 20 Test
train_years_hitters = range(2000, 2019)
test_years_hitters = range(2019, 2025)

train_years_pitchers = range(2000, 2019)
test_years_pitchers = range(2019, 2025)

train_hitters = HD[HD['Season'].isin(train_years_hitters)]
test_hitters = HD[HD['Season'].isin(test_years_hitters)]

train_pitchers = PD[PD['Season'].isin(train_years_pitchers)]
test_pitchers = PD[PD['Season'].isin(test_years_pitchers)]



# Hitter stats for next season
y_hitters = ['WAR_next', 'H_next', 'HR_next', 'RBI_next', 'BB%_next', 'K%_next',
             'AVG_next', 'SLG_next']

# Drop rows with NaN in target columns first
train_hitters = train_hitters.dropna(subset=y_hitters)
test_hitters = test_hitters.dropna(subset=y_hitters)

# Then drop rows with NaN in ANY feature (cleaner data, better predictions)
train_hitters = train_hitters.dropna()
test_hitters = test_hitters.dropna()


features_to_remove = ['G', 'AB', 'R', 'RBI', 'SB', 'H',  
                      'AVG_norm', 'OBP_norm', 'SLG_norm']

X_train_hitters = train_hitters.drop(columns=['IDfg', 'Name', 'Season', 'Played_next_year'] + y_hitters + features_to_remove)
y_train_hitters = train_hitters[y_hitters]

X_test_hitters = test_hitters.drop(columns=['IDfg', 'Name', 'Season', 'Played_next_year'] + y_hitters + features_to_remove)
y_test_hitters = test_hitters[y_hitters]


scale_hitters = MinMaxScaler()
X_train_hitters_scaled = scale_hitters.fit_transform(X_train_hitters)
X_test_hitters_scaled = scale_hitters.transform(X_test_hitters)


# Pitcher stats for next season
y_pitchers = ['WAR_next', 'W_next', 'L_next', 'IP_next', 'ER_next', 'SO_next',
             'K%_next', 'BB%_next', 'HR/9_next', 'WHIP_next','ERA_next']

# Drop rows with NaN in target columns
train_pitchers = train_pitchers.dropna(subset=y_pitchers)
test_pitchers = test_pitchers.dropna(subset=y_pitchers)


train_pitchers = train_pitchers.dropna()
test_pitchers = test_pitchers.dropna()


pitcher_features_to_remove = ['G', 'GS', 'W', 'L', 'H', 'ER', 'BB', 'SO']  # Keep HR, WHIP, ERA 

X_train_pitchers = train_pitchers.drop(columns=['IDfg','Name', 'Season', 'Played_next_year'] + y_pitchers + pitcher_features_to_remove)
y_train_pitchers = train_pitchers[y_pitchers]

X_test_pitchers = test_pitchers.drop(columns=['IDfg','Name', 'Season', 'Played_next_year'] + y_pitchers + pitcher_features_to_remove)
y_test_pitchers = test_pitchers[y_pitchers]

scale_pitchers = MinMaxScaler()
X_train_pitchers_scaled = scale_pitchers.fit_transform(X_train_pitchers)
X_test_pitchers_scaled = scale_pitchers.transform(X_test_pitchers)

# ==========================================================================

# ==========================================================================
# Train Models (Step 4)

# Training the Hitter Model
hitter = MultiOutputRegressor(
    xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=800,           
        learning_rate=0.03,         
        max_depth=7,                
        subsample=0.85,             
        colsample_bytree=0.85,      
        min_child_weight=1,         
        gamma=0.05,                 
        reg_alpha=0.1,              
        reg_lambda=1.0,             
        random_state=42
    )
)
hitter.fit(X_train_hitters_scaled, y_train_hitters)
pickle.dump(hitter, open('hitter_model.pkl', 'wb'))
pickle.dump(scale_hitters, open('hitter_scaler.pkl', 'wb'))
accuracy = hitter.score(X_test_hitters_scaled, y_test_hitters)
print(f"Hitter Model Accuracy (Overall): {accuracy:.2f}")


y_pred_hitters = hitter.predict(X_test_hitters_scaled)
war_accuracy_h = metrics.r2_score(y_test_hitters['WAR_next'], y_pred_hitters[:, 0])


mae_h = metrics.mean_absolute_error(y_test_hitters['WAR_next'], y_pred_hitters[:, 0])
mse_h = metrics.mean_squared_error(y_test_hitters['WAR_next'], y_pred_hitters[:, 0])
rmse_h = np.sqrt(mse_h)
mape_h = np.mean(np.abs((y_test_hitters['WAR_next'] - y_pred_hitters[:, 0]) / y_test_hitters['WAR_next'])) * 100

print(f"Hitter WAR_next R² Score: {war_accuracy_h:.3f}")
print(f"Hitter WAR_next MAE: {mae_h:.3f} WAR")
print(f"Hitter WAR_next RMSE: {rmse_h:.3f} WAR")
print(f"Hitter WAR_next MAPE: {mape_h:.2f}%")


pitcher = MultiOutputRegressor(
    xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=600,           
        learning_rate=0.03,         
        max_depth=7,                
        subsample=0.85,             
        colsample_bytree=0.85,      
        min_child_weight=2,         
        gamma=0.05,                 
        reg_alpha=0.1,              
        reg_lambda=1.0,             
        random_state=42
    )
)
pitcher.fit(X_train_pitchers_scaled, y_train_pitchers)
pickle.dump(pitcher, open('pitcher_model.pkl', 'wb'))
pickle.dump(scale_pitchers, open('pitcher_scaler.pkl', 'wb'))
accuracy = pitcher.score(X_test_pitchers_scaled, y_test_pitchers)
print(f"Pitcher Model Accuracy (Overall): {accuracy:.2f}")

# Get per-stat accuracy and comprehensive metrics
y_pred_pitchers = pitcher.predict(X_test_pitchers_scaled)
war_accuracy_p = metrics.r2_score(y_test_pitchers['WAR_next'], y_pred_pitchers[:, 0])

# Calculate all metrics for WAR_next
mae_p = metrics.mean_absolute_error(y_test_pitchers['WAR_next'], y_pred_pitchers[:, 0])
mse_p = metrics.mean_squared_error(y_test_pitchers['WAR_next'], y_pred_pitchers[:, 0])
rmse_p = np.sqrt(mse_p)
mape_p = np.mean(np.abs((y_test_pitchers['WAR_next'] - y_pred_pitchers[:, 0]) / y_test_pitchers['WAR_next'])) * 100

# ==========================================================================
# Basic Neural Network Model (Step 4 continued)

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
EarlyStopping = tf.keras.callbacks.EarlyStopping


hitter_nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_hitters_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(y_hitters), activation='linear')
])

# Use a lower learning rate for better convergence
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
hitter_nn.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

hitter_nn.fit(X_train_hitters_scaled, y_train_hitters, 
              epochs=500, batch_size=16, validation_split=0.2, 
              callbacks=[early_stop], verbose=0)
hitter_nn.save('hitter_nn.h5')

y_pred_hitters_nn = hitter_nn.predict(X_test_hitters_scaled, verbose=0)
r2_nn_h = metrics.r2_score(y_test_hitters, y_pred_hitters_nn)
r2_war_nn_h = metrics.r2_score(y_test_hitters['WAR_next'], y_pred_hitters_nn[:, 0])

# Calculate all metrics for NN
mae_nn_h = metrics.mean_absolute_error(y_test_hitters['WAR_next'], y_pred_hitters_nn[:, 0])
mse_nn_h = metrics.mean_squared_error(y_test_hitters['WAR_next'], y_pred_hitters_nn[:, 0])
rmse_nn_h = np.sqrt(mse_nn_h)
mape_nn_h = np.mean(np.abs((y_test_hitters['WAR_next'] - y_pred_hitters_nn[:, 0]) / y_test_hitters['WAR_next'])) * 100


pitcher_nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_pitchers_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(y_pitchers), activation='linear')
])

optimizer_p = tf.keras.optimizers.Adam(learning_rate=0.001)
pitcher_nn.compile(optimizer=optimizer_p, loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stop_p = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

pitcher_nn.fit(X_train_pitchers_scaled, y_train_pitchers,
               epochs=500, batch_size=16, validation_split=0.2,
               callbacks=[early_stop_p], verbose=0)
pitcher_nn.save('pitcher_nn.h5')

y_pred_pitchers_nn = pitcher_nn.predict(X_test_pitchers_scaled, verbose=0)
r2_nn_p = metrics.r2_score(y_test_pitchers, y_pred_pitchers_nn)
r2_war_nn_p = metrics.r2_score(y_test_pitchers['WAR_next'], y_pred_pitchers_nn[:, 0])

# Calculate all metrics for NN
mae_nn_p = metrics.mean_absolute_error(y_test_pitchers['WAR_next'], y_pred_pitchers_nn[:, 0])
mse_nn_p = metrics.mean_squared_error(y_test_pitchers['WAR_next'], y_pred_pitchers_nn[:, 0])
rmse_nn_p = np.sqrt(mse_nn_p)
mape_nn_p = np.mean(np.abs((y_test_pitchers['WAR_next'] - y_pred_pitchers_nn[:, 0]) / y_test_pitchers['WAR_next'])) * 100
# ==========================================================================

# ==========================================================================
# Evaluate Models (Visualizations and Metrics) (Step 5)

print(f"Hitter WAR_next R² Score: {war_accuracy_h:.3f}")
print(f"Hitter WAR_next MAE: {mae_h:.3f} WAR")
print(f"Hitter WAR_next RMSE: {rmse_h:.3f} WAR")
print(f"Hitter WAR_next MAPE: {mape_h:.2f}%")

print(f"Pitcher WAR_next R² Score: {war_accuracy_p:.3f}")
print(f"Pitcher WAR_next MAE: {mae_p:.3f} WAR")
print(f"Pitcher WAR_next RMSE: {rmse_p:.3f} WAR")
print(f"Pitcher WAR_next MAPE: {mape_p:.2f}%")

print(f"Hitter NN WAR_next R² Score: {r2_war_nn_h:.3f}")
print(f"Hitter NN WAR_next MAE: {mae_nn_h:.3f} WAR")
print(f"Hitter NN WAR_next RMSE: {rmse_nn_h:.3f} WAR")
print(f"Hitter NN WAR_next MAPE: {mape_nn_h:.2f}%")

print(f"Pitcher NN WAR_next R² Score: {r2_war_nn_p:.3f}")
print(f"Pitcher NN WAR_next MAE: {mae_nn_p:.3f} WAR")
print(f"Pitcher NN WAR_next RMSE: {rmse_nn_p:.3f} WAR")
print(f"Pitcher NN WAR_next MAPE: {mape_nn_p:.2f}%")

# Hitter Predicted vs Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_hitters['WAR_next'], y_pred_hitters[:, 0], alpha=0.5)
plt.plot([y_test_hitters['WAR_next'].min(), y_test_hitters['WAR_next'].max()], 
         [y_test_hitters['WAR_next'].min(), y_test_hitters['WAR_next'].max()], 'r--')
plt.xlabel('Actual WAR')
plt.ylabel('Predicted WAR')
plt.title('Hitter XGBoost: Predicted vs Actual WAR')
plt.savefig('Hitter_Predicted_vs_Actual_WAR.png')
plt.show()

# Pitcher Predicted vs Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pitchers['WAR_next'], y_pred_pitchers[:, 0], alpha=0.5)
plt.plot([y_test_pitchers['WAR_next'].min(), y_test_pitchers['WAR_next'].max()], 
         [y_test_pitchers['WAR_next'].min(), y_test_pitchers['WAR_next'].max()], 'r--')
plt.xlabel('Actual WAR')
plt.ylabel('Predicted WAR')
plt.title('Pitcher XGBoost: Predicted vs Actual WAR')
plt.savefig('Pitcher_Predicted_vs_Actual_WAR.png')
plt.show()


# ==========================================================================