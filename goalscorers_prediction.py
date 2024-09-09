import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

# Load datasets
matchday_data = pd.read_csv('2023_matchday_results.csv')
shooting_data = pd.read_csv('player_premier_league_shooting.csv')
stats_data = pd.read_csv('player_premier_league_stats.csv')
passing_data = pd.read_csv('player_premier_league_passing.csv')

# Function to get players from a specific team
def get_team_players(team_name, dataset):
    return dataset[dataset['Squad'] == team_name].copy()

# Function to calculate features for each dataset
def calculate_features(df, feature_type):
    if feature_type == 'shooting':
        df['goal_per_shot'] = df['Goals'] / (df['Tot_Shot'] + 1)
        df['goal_prob'] = df['xG'] + df['goal_per_shot']
    elif feature_type == 'stats':
        df['scoring_impact'] = df['Goals'] + df['xG'] + df['Assist']
    elif feature_type == 'passing':
        df['passing_impact'] = df['KeyPas'] + df['ProgPass']
    return df

# Function to normalize features
def scale_features(df, feature_column, scaler):
    df[f'{feature_column}_scaled'] = scaler.fit_transform(df[[feature_column]].fillna(0))
    return df

# Function to predict goalscorers for a match using independent datasets
def predict_goalscorers(match, shooting_data, stats_data, passing_data):
    home_team = match['teams.home.name']
    away_team = match['teams.away.name']
    home_goals = match['goals.home']
    away_goals = match['goals.away']

    # Get home and away players independently from different datasets
    home_shooting = get_team_players(home_team, shooting_data)
    away_shooting = get_team_players(away_team, shooting_data)

    home_stats = get_team_players(home_team, stats_data)
    away_stats = get_team_players(away_team, stats_data)

    home_passing = get_team_players(home_team, passing_data)
    away_passing = get_team_players(away_team, passing_data)

    # Impute missing data and calculate features for shooting, stats, and passing datasets
    datasets = [(home_shooting, 'shooting'), (away_shooting, 'shooting'),
                (home_stats, 'stats'), (away_stats, 'stats'),
                (home_passing, 'passing'), (away_passing, 'passing')]

    imputer = SimpleImputer(strategy='mean')
    for df, feature_type in datasets:
        if not df.empty:
            # Impute relevant columns
            for col in ['Goals', 'Tot_Shot', 'xG', 'Assist', 'KeyPas', 'ProgPass']:
                if col in df:
                    df[col] = imputer.fit_transform(df[[col]])

            # Calculate features based on type
            df = calculate_features(df, feature_type)

    # Normalize features
    scaler = StandardScaler()
    if not home_shooting.empty:
        home_shooting = scale_features(home_shooting, 'goal_prob', scaler)
    if not away_shooting.empty:
        away_shooting = scale_features(away_shooting, 'goal_prob', scaler)

    if not home_stats.empty:
        home_stats = scale_features(home_stats, 'scoring_impact', scaler)
    if not away_stats.empty:
        away_stats = scale_features(away_stats, 'scoring_impact', scaler)

    if not home_passing.empty:
        home_passing = scale_features(home_passing, 'passing_impact', scaler)
    if not away_passing.empty:
        away_passing = scale_features(away_passing, 'passing_impact', scaler)

    # Function to aggregate player data and calculate final scores
    def aggregate_player_scores(shooting, stats, passing, goals):
        if not shooting.empty:
            players = shooting[['Player', 'goal_prob_scaled']].copy()
            if not stats.empty:
                players['scoring_impact_scaled'] = stats['scoring_impact_scaled'].values
            if not passing.empty:
                players['passing_impact_scaled'] = passing['passing_impact_scaled'].values

            players['final_score'] = (players['goal_prob_scaled'] +
                                      players.get('scoring_impact_scaled', 0) +
                                      players.get('passing_impact_scaled', 0)) / 3
            top_scorers = players.sort_values(by='final_score', ascending=False).head(goals)
            return top_scorers['Player'].tolist()
        return ["Unknown Scorer"] * goals

    # Calculate top scorers for both home and away teams
    home_scorers = aggregate_player_scores(home_shooting, home_stats, home_passing, home_goals)
    away_scorers = aggregate_player_scores(away_shooting, away_stats, away_passing, away_goals)

    return home_scorers, away_scorers


# Mapping team names from 2023_matchday_results.csv to team names in player datasets
team_name_mapping = {
    'Leeds': 'Leeds United',
    'Newcastle': 'Newcastle Utd',
    'Manchester United': 'Manchester Utd',
    'Nottingham Forest': "Nott'ham Forest",
    'Leicester': 'Leicester City',
}

# Replace team names in the matchday data using the mapping
matchday_data['teams.home.name'] = matchday_data['teams.home.name'].replace(team_name_mapping)
matchday_data['teams.away.name'] = matchday_data['teams.away.name'].replace(team_name_mapping)

# Iterate through each match and predict goalscorers
for idx, row in matchday_data.iterrows():
    home_team = row['teams.home.name']
    away_team = row['teams.away.name']
    home_goals = int(row['goals.home'])
    away_goals = int(row['goals.away'])

    # Use the new prediction function to get top predicted scorers
    if home_goals > 0 or away_goals > 0:
        home_scorers, away_scorers = predict_goalscorers(row, shooting_data, stats_data, passing_data)
    else:
        home_scorers, away_scorers = [], []

    # Fix for missing goalscorers
    if home_goals > 0 and not home_scorers:
        print(f"Warning: No available players for {home_team}, randomly selecting scorers.")
        home_scorers = ["Unknown Scorer"] * home_goals

    if away_goals > 0 and not away_scorers:
        print(f"Warning: No available players for {away_team}, randomly selecting scorers.")
        away_scorers = ["Unknown Scorer"] * away_goals
    
    # Print results for the match, handle zero-goal cases
    home_scorer_list = ', '.join(home_scorers) if home_goals > 0 else 'None'
    away_scorer_list = ', '.join(away_scorers) if away_goals > 0 else 'None'
    
    print(f"Match: {home_team} vs {away_team}")
    print(f"Home Goals: {home_goals} - Scorers: {home_scorer_list}")
    print(f"Away Goals: {away_goals} - Scorers: {away_scorer_list}\n")
