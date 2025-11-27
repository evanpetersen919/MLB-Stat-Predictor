import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import pickle
import os

# Page config
st.set_page_config(page_title="MLB Statistics Predictor", layout='centered')

# ========================================================================
# Data Loading Function
# ========================================================================
@st.cache_data
def load_data():
    # Load CSV files
    HD = pd.read_csv('hitters_war_dataset_enhanced.csv')
    PD = pd.read_csv('pitchers_war_dataset_enhanced.csv')
    
    # Sort by player ID and season to maintain chronological order
    HD = HD.sort_values(['IDfg', 'Season']).reset_index(drop=True)
    PD = PD.sort_values(['IDfg', 'Season']).reset_index(drop=True)
    
    return HD, PD

# ========================================================================
# Hitter Model Loading Function
# ========================================================================
@st.cache_data
def prepare_and_train_hitters(HD):
    
    # Define target variables to predict for next season
    y_hitters = ['WAR_next', 'H_next', 'HR_next', 'RBI_next', 'BB%_next', 'K%_next',
                 'AVG_next', 'SLG_next']

    # Split data by season: 2000-2018 for training, 2019-2024 for testing (80/20 split)
    train_years_hitters = range(2000, 2019)
    test_years_hitters = range(2019, 2025)

    train_hitters = HD[HD['Season'].isin(train_years_hitters)].copy()
    test_hitters = HD[HD['Season'].isin(test_years_hitters)].copy()

    # Remove rows with missing target values or any NA values for clean training data
    train_hitters = train_hitters.dropna(subset=y_hitters).dropna()
    test_hitters = test_hitters.dropna(subset=y_hitters).dropna()

    # Remove features that would cause data leakage or are redundant
    features_to_remove = ['G', 'AB', 'R', 'RBI', 'SB', 'H', 'AVG_norm', 'OBP_norm', 'SLG_norm']
    drop_cols = ['IDfg', 'Name', 'Season', 'Played_next_year'] + y_hitters + features_to_remove

    # Create feature matrix (X) and target matrix (y) - needed to get feature names and test data
    X_train = train_hitters.drop(columns=[c for c in drop_cols if c in train_hitters.columns])
    X_test = test_hitters.drop(columns=[c for c in drop_cols if c in test_hitters.columns])
    y_test = test_hitters[y_hitters]

    # Load pre-trained model and scaler from training script
    import pickle
    with open('hitter_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('hitter_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler, X_train.columns.tolist(), y_hitters, X_test, y_test
# ========================================================================


# ========================================================================
# Pitcher Model Loading Function
# ========================================================================
@st.cache_data
def prepare_and_train_pitchers(PD):

    # Define target variables to predict for next season
    y_pitchers = ['WAR_next', 'W_next', 'L_next', 'IP_next', 'ER_next', 'SO_next', 'K%_next', 'BB%_next', 'HR/9_next', 'WHIP_next', 'ERA_next']
    
    # Split data by season: 2000-2018 for training, 2019-2024 for testing
    train_pitchers = PD[PD['Season'].isin(range(2000, 2019))].copy()
    test_pitchers = PD[PD['Season'].isin(range(2019, 2025))].copy()
    
    # Remove rows with missing values
    train_pitchers = train_pitchers.dropna(subset=y_pitchers).dropna()
    test_pitchers = test_pitchers.dropna(subset=y_pitchers).dropna()
    
    # Remove features that would cause data leakage or are redundant
    pitcher_features_to_remove = ['G', 'GS', 'W', 'L', 'H', 'ER', 'BB', 'SO']
    drop_cols = ['IDfg', 'Name', 'Season', 'Played_next_year'] + y_pitchers + pitcher_features_to_remove
    
    # Create feature matrix (X) and target matrix (y)
    X_train = train_pitchers.drop(columns=[c for c in drop_cols if c in train_pitchers.columns])
    X_test = test_pitchers.drop(columns=[c for c in drop_cols if c in test_pitchers.columns])
    y_test = test_pitchers[y_pitchers]
    
    # Load pre-trained model and scaler from training script
    model = pickle.load(open('pitcher_model.pkl', 'rb'))
    scaler = pickle.load(open('pitcher_scaler.pkl', 'rb'))
    
    return model, scaler, X_train.columns.tolist(), y_pitchers, X_test, y_test
# ========================================================================

# ========================================================================
# Main Streamlit App
# ========================================================================
def main():
    st.title('MLB Stat Predictor')
    st.markdown('**XGBoost Machine Learning Model** | R² = 0.829 (Hitters) | R² = 0.735 (Pitchers)')
    st.markdown('---')

    # Player type selector (Hitters or Pitchers)
    player_type = st.radio('Select Player Type:', ['Hitters', 'Pitchers'], horizontal=True)

    # Load datasets
    HD, PD = load_data()

    # Load appropriate model and data based on player type selection
    if player_type == 'Hitters':
        model, scaler, feature_names, y_cols, X_test, y_test = prepare_and_train_hitters(HD)
        dataset = HD
    else:
        model, scaler, feature_names, y_cols, X_test, y_test = prepare_and_train_pitchers(PD)
        dataset = PD

    # Two prediction modes
    st.markdown('### Choose Prediction Mode')
    col1, col2 = st.columns(2)
    with col1:
        # Mode 1: Create custom player with manual stat input
        if st.button('Create Your Player', type='primary', use_container_width=True):
            st.session_state['mode'] = 'create'
    with col2:
        # Mode 2: Select existing player from dataset
        if st.button('Predict Favorite Player', type='primary', use_container_width=True):
            st.session_state['mode'] = 'favorite'

    st.markdown('---')

    # Get current mode from session state
    mode = st.session_state.get('mode', None)

# ========================================================================
# Mode 1: Create Your Player (Manual Stat Input)
# ========================================================================

    if mode == 'create':
        st.markdown(f'### Create Your {player_type[:-1]}')
        st.write('Enter current season stats to predict next season performance.')
        st.write('')

        # Dictionary to hold user inputs
        inputs = {}

        if player_type == 'Hitters':
            common = ['Age', 'PA', 'BB%', 'K%', 'ISO', 'wOBA', 'wRC+', 'WAR', 'HR', 'AVG', 'OBP', 'SLG']
        else:  # Pitchers
            common = ['Age', 'IP', 'K%', 'BB%', 'K-BB%', 'WAR', 'HR/9', 'WHIP', 'FIP', 'ERA']

        available = [c for c in common if c in feature_names]

        st.markdown('**Key Statistics:**')

        for c in available:
            # Use mean as default value
            default_val = float(HD[c].mean()) if c in HD.columns else 0.0
            inputs[c] = st.number_input(f'{c}', value=default_val)

        # Get remaining features not in the key statistics list
        extra_feats = [f for f in feature_names if f not in available]
        
        # Optional: Show advanced features
        show_advanced = st.checkbox('Show all features', value=False)
        if show_advanced:
            st.write('**Additional Features:**')
            for f in extra_feats:
                if f in dataset.columns and pd.api.types.is_numeric_dtype(dataset[f]):
                    default_val = float(dataset[f].mean())
                    inputs[f] = st.number_input(f, value=default_val)

        # Generate prediction when button is clicked
        if st.button('Predict Next Season Stats'):
            # Build feature vector in the exact order expected by the model
            row = []
            for f in feature_names:
                if f in inputs:
                    row.append(inputs[f])  # Use user input if available
                else:
                    # Use dataset median for features not provided by user
                    row.append(float(np.round(dataset[f].median() if f in dataset.columns else 0.0, 3)))
            
            # Scale features and make prediction
            x = scaler.transform([row])
            preds = model.predict(x)[0]  # Get predictions for all targets
            out = {col: float(np.round(val, 3)) for col, val in zip(y_cols, preds)}

            # Display predicted statistics in a grid
            st.write('')
            st.markdown('**Predicted Next Season Statistics:**')
            cols = st.columns(4)
            for idx, (k, v) in enumerate(out.items()):
                with cols[idx % 4]:
                    st.metric(label=k, value=v)
            
            # Calculate and display performance percentile based on WAR
            war = out.get('WAR_next', None)
            if war is not None:
                hist = y_test['WAR_next']  # Historical WAR values from test set
                pct = float((hist < war).mean())  # Calculate percentile
                st.write('')
                st.markdown(f'**Performance Rating:** {int(pct*100)}th percentile')
                st.caption('Compared to 2019-2024 MLB players')
                st.progress(min(max(pct, 0.0), 1.0))  # Visual progress bar

    # ====================================================================
    # Mode 2: Predict Favorite Player (Select from Dataset)
    # ====================================================================
    elif mode == 'favorite':
        st.markdown(f'### Predict Favorite {player_type[:-1]} Stats')
        st.write('Select a player from the dataset to predict next season performance.')
        st.write('')
        
        # Get list of all unique player names from dataset
        names = dataset['Name'].dropna().unique().tolist()
        name = st.selectbox('Select Player', options=[''] + sorted(names))
        
        # Generate prediction when player is selected and button clicked
        if name and st.button('Generate Prediction', type='primary'):
            # Get most recent season data for selected player
            player_row = dataset[dataset['Name'] == name].sort_values('Season').iloc[-1:]
            
            if player_row.empty:
                st.error('Player not found in dataset')
            else:
                season = player_row.iloc[0]['Season']
                st.write('')
                st.markdown(f'**Player:** {name} | **Last Season:** {int(season)}')
                st.write('')
                
                # Remove non-feature columns to create feature vector
                X_row = player_row.drop(columns=[c for c in ['IDfg', 'Name', 'Season', 'Played_next_year'] if c in player_row.columns])
                
                # Build feature vector in the exact order expected by the model
                row = []
                for f in feature_names:
                    if f in X_row.columns:
                        row.append(float(X_row.iloc[0][f]))  # Use player's actual stat
                    else:
                        # Use dataset median for missing features
                        row.append(float(np.round(dataset[f].median() if f in dataset.columns else 0.0, 3)))
                
                # Scale features and make prediction
                x = scaler.transform([row])
                preds = model.predict(x)[0]  # Get predictions for all targets
                out = {col: float(np.round(val, 3)) for col, val in zip(y_cols, preds)}
                
                # Display predicted statistics in a grid
                st.markdown('**Predicted Next Season Statistics:**')
                cols = st.columns(4)
                for idx, (k, v) in enumerate(out.items()):
                    with cols[idx % 4]:
                        st.metric(label=k, value=v)
                
                # Calculate and display performance percentile based on WAR
                war = out.get('WAR_next', None)
                if war is not None:
                    hist = y_test['WAR_next']  # Historical WAR values from test set
                    pct = float((hist < war).mean())  # Calculate percentile
                    st.write('')
                    st.markdown(f'**Performance Rating:** {int(pct*100)}th percentile')
                    st.caption('Compared to 2019-2024 MLB players')
                    st.progress(min(max(pct, 0.0), 1.0))  # Visual progress bar
    
    # ========================================================================
    # No Mode Selected
    # ========================================================================
    else:
        st.info('Select a mode above to get started')

    # ========================================================================
    # Footer
    # ========================================================================
    st.write('---')
    if player_type == 'Hitters':
        st.caption('Model: XGBoost MultiOutputRegressor (hitter_model.pkl) | Trained on 2000-2018 data, tested on 2019-2024')
    else:
        st.caption('Model: XGBoost MultiOutputRegressor (pitcher_model.pkl) | Trained on 2000-2018 data, tested on 2019-2024')

    st.write('')
    st.caption('Developed by Evan Petersen & Michael Hernandez')
# ========================================================================
# Run Application
# ========================================================================
if __name__ == '__main__':
    main()