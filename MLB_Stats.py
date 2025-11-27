from pybaseball import batting_stats, pitching_stats
import pandas as pd
import numpy as np

START_YEAR = 2000
END_YEAR = 2025
MIN_PA = 200
MIN_IP = 50


def add_career_features(df, id_col):
    df = df.sort_values([id_col, 'Season'])
    df['Career_Seasons'] = df.groupby(id_col).cumcount() + 1
    df['WAR_lag1'] = df.groupby(id_col)['WAR'].shift(1)
    df['WAR_lag2'] = df.groupby(id_col)['WAR'].shift(2)
    df['WAR_lag3'] = df.groupby(id_col)['WAR'].shift(3)
    df['WAR_trend'] = df['WAR_lag1'] - df['WAR_lag2']
    df['WAR_rolling_2yr'] = df.groupby(id_col)['WAR_lag1'].rolling(window=2, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Career_WAR_avg'] = df.groupby(id_col)['WAR_lag1'].expanding().mean().reset_index(level=0, drop=True)
    df['Age_squared'] = df['Age'] ** 2
    df['Prime_years'] = ((df['Age'] >= 27) & (df['Age'] <= 30)).astype(int)
    return df


def build_next_year_pairs_hitters(df, id_col):
    df = df.sort_values([id_col, 'Season'])
    df['WAR_next'] = df.groupby(id_col)['WAR'].shift(-1)
    df['H_next'] = df.groupby(id_col)['H'].shift(-1)
    df['HR_next'] = df.groupby(id_col)['HR'].shift(-1)
    df['RBI_next'] = df.groupby(id_col)['RBI'].shift(-1)
    df['BB%_next'] = df.groupby(id_col)['BB%'].shift(-1)
    df['K%_next'] = df.groupby(id_col)['K%'].shift(-1)
    df['AVG_next'] = df.groupby(id_col)['AVG'].shift(-1)
    df['OBP_next'] = df.groupby(id_col)['OBP'].shift(-1)
    df['SLG_next'] = df.groupby(id_col)['SLG'].shift(-1)
    df['wOBA_next'] = df.groupby(id_col)['wOBA'].shift(-1)
    df['wRC+_next'] = df.groupby(id_col)['wRC+'].shift(-1)
    df['Played_next_year'] = (~df['WAR_next'].isna()).astype(int)
    df = df.dropna(subset=['WAR_next'])
    return df

def build_next_year_pairs_pitchers(df, id_col):
    df = df.sort_values([id_col, 'Season'])
    df['WAR_next'] = df.groupby(id_col)['WAR'].shift(-1)
    df['W_next'] = df.groupby(id_col)['W'].shift(-1)
    df['L_next'] = df.groupby(id_col)['L'].shift(-1)
    df['IP_next'] = df.groupby(id_col)['IP'].shift(-1)
    df['ER_next'] = df.groupby(id_col)['ER'].shift(-1)
    df['SO_next'] = df.groupby(id_col)['SO'].shift(-1)
    df['K%_next'] = df.groupby(id_col)['K%'].shift(-1)
    df['BB%_next'] = df.groupby(id_col)['BB%'].shift(-1)
    df['K-BB%_next'] = df.groupby(id_col)['K-BB%'].shift(-1)
    df['HR/9_next'] = df.groupby(id_col)['HR/9'].shift(-1)
    df['WHIP_next'] = df.groupby(id_col)['WHIP'].shift(-1)
    df['FIP_next'] = df.groupby(id_col)['FIP'].shift(-1)
    df['ERA_next'] = df.groupby(id_col)['ERA'].shift(-1)
    df['Played_next_year'] = (~df['WAR_next'].isna()).astype(int)
    df = df.dropna(subset=['WAR_next'])
    return df
   

# HITTERS
hitter_frames = []
for year in range(START_YEAR, END_YEAR + 1):
    try:
        stats = batting_stats(year, qual=0)
        stats['Season'] = year
        hitter_frames.append(stats)
    except:
        pass

hitters = pd.concat(hitter_frames, ignore_index=True)
id_col = 'playerid' if 'playerid' in hitters.columns else 'IDfg'
hitters = hitters[hitters['PA'] >= MIN_PA]

hitter_cols = [id_col, 'Name', 'Season', 'Age', 'G', 'PA', 'AB', 'H', 'HR', 'R', 'RBI', 'SB',
               'BB%', 'K%', 'ISO', 'BABIP', 'AVG', 'OBP', 'SLG', 'wOBA', 'wRC+', 
               'Off', 'Def', 'WAR']
hitter_cols = [col for col in hitter_cols if col in hitters.columns]
hitters = hitters[hitter_cols].dropna()
hitters = hitters[hitters['Season'] != 2020]

for stat in ['HR', 'AVG', 'OBP', 'SLG', 'ISO', 'wOBA']:
    if stat in hitters.columns:
        league_mean = hitters.groupby('Season')[stat].transform('mean')
        league_std = hitters.groupby('Season')[stat].transform('std')
        hitters[f'{stat}_norm'] = (hitters[stat] - league_mean) / (league_std + 0.001)

hitters = add_career_features(hitters, id_col)
hitters_dataset = build_next_year_pairs_hitters(hitters, id_col)

if 'BB%' in hitters_dataset.columns and 'K%' in hitters_dataset.columns:
    hitters_dataset['BB_K_ratio'] = hitters_dataset['BB%'] / (hitters_dataset['K%'] + 0.001)
if 'PA' in hitters_dataset.columns:
    hitters_dataset['Full_time'] = (hitters_dataset['PA'] >= 500).astype(int)
if 'G' in hitters_dataset.columns:
    hitters_dataset['Availability'] = hitters_dataset['G'] / 162

hitters_dataset.to_csv("hitters_war_dataset_enhanced.csv", index=False)


# PITCHERS
pitcher_frames = []
for year in range(START_YEAR, END_YEAR + 1):
    try:
        stats = pitching_stats(year, qual=0)
        stats['Season'] = year
        pitcher_frames.append(stats)
    except:
        pass

pitchers = pd.concat(pitcher_frames, ignore_index=True)
id_col_p = 'playerid' if 'playerid' in pitchers.columns else 'IDfg'
pitchers = pitchers[pitchers['IP'] >= MIN_IP]

pitcher_cols = [id_col_p, 'Name', 'Season', 'Age', 'W', 'L', 'G', 'GS', 'IP', 'H', 'ER', 'HR', 
                'BB', 'SO', 'K%', 'BB%', 'K-BB%', 'HR/9', 'WHIP', 'FIP', 'xFIP', 'ERA', 'WAR']
pitcher_cols = [col for col in pitcher_cols if col in pitchers.columns]
pitchers = pitchers[pitcher_cols].dropna()
pitchers = pitchers[pitchers['Season'] != 2020]

for stat in ['ERA', 'FIP', 'xFIP', 'K%', 'BB%', 'HR/9']:
    if stat in pitchers.columns:
        league_mean = pitchers.groupby('Season')[stat].transform('mean')
        league_std = pitchers.groupby('Season')[stat].transform('std')
        pitchers[f'{stat}_norm'] = (pitchers[stat] - league_mean) / (league_std + 0.001)

pitchers = add_career_features(pitchers, id_col_p)
pitchers_dataset = build_next_year_pairs_pitchers(pitchers, id_col_p)

if 'GS' in pitchers_dataset.columns and 'G' in pitchers_dataset.columns:
    pitchers_dataset['Starter_pct'] = pitchers_dataset['GS'] / (pitchers_dataset['G'] + 0.001)
    pitchers_dataset['Is_Starter'] = (pitchers_dataset['Starter_pct'] >= 0.5).astype(int)
if 'IP' in pitchers_dataset.columns:
    pitchers_dataset['Heavy_workload'] = (pitchers_dataset['IP'] >= 180).astype(int)
if 'G' in pitchers_dataset.columns and 'GS' in pitchers_dataset.columns:
    expected_games = pitchers_dataset['GS'] * 1.0 + (pitchers_dataset['G'] - pitchers_dataset['GS']) * 0.3
    pitchers_dataset['Availability'] = pitchers_dataset['G'] / (expected_games + 1)

pitchers_dataset.to_csv("pitchers_war_dataset_enhanced.csv", index=False)
