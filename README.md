# MLB Stat Predictor

A machine learning project that predicts next season MLB player performance using XGBoost and Neural Networks.

## Authors
- **Evan Petersen**
- **Michael Hernandez**

Developed for CS110 Final Project, December 2025

## Project Overview

This project uses historical MLB data (2000-2025) to predict future player statistics, achieving:
- **Hitters:** R² = 0.829 (XGBoost)
- **Pitchers:** R² = 0.735 (XGBoost)

## Features

- **Two Machine Learning Models:**
  - XGBoost MultiOutputRegressor (primary model)
  - Neural Network with dropout and batch normalization
  
- **Interactive Streamlit Web App:**
  - Create custom player predictions
  - Select existing players from dataset
  - Real-time performance predictions

- **Predicted Statistics:**
  - **Hitters:** WAR, HR, RBI, AVG, OBP, SLG, wOBA, wRC+
  - **Pitchers:** WAR, W, L, IP, ERA, WHIP, FIP, K%, BB%

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

### Train Models
```bash
python MLB.py
```

### Data Collection
```bash
python MLB_Stats.py
```

## Files

- `MLB.py` - Main training script with XGBoost and Neural Network models
- `streamlit_app.py` - Interactive web interface
- `MLB_Stats.py` - Data collection from FanGraphs
- `hitters_war_dataset_enhanced.csv` - Hitter statistics dataset
- `pitchers_war_dataset_enhanced.csv` - Pitcher statistics dataset
- `*.pkl` - Saved XGBoost models and scalers
- `*.h5` - Saved Neural Network models

## Model Performance

| Model | Hitter R² | Pitcher R² |
|-------|-----------|------------|
| XGBoost | 0.829 | 0.735 |
| Neural Network | 0.635 | 0.547 |

## Live Demo

[Streamlit App](https://mlb-stat-predictor-kjsuvrdjedxnmzafgecv5g.streamlit.app/)

## Technologies

- Python 3.10
- XGBoost
- TensorFlow/Keras
- Scikit-learn
- Streamlit
- Pandas, NumPy, Matplotlib

## License

MIT License
