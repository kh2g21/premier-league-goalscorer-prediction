import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import random

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
        df['goal_prob'] = 0.8 * df['Goals'] + 0.45 * df['xG']  # Weight more on actual goals
    elif feature_type == 'stats':
        df['scoring_impact'] = 0.8 * df['Goals'] + 0.6 * df['xG'] + 0.6 * df['Assist']
    elif feature_type == 'passing':
        df['passing_impact'] = df['KeyPas'] + df['ProgPass']
    return df

# Function to normalize features
def scale_features(df, feature_column, scaler):
    df[f'{feature_column}_scaled'] = scaler.fit_transform(df[[feature_column]].fillna(0))
    return df

def assign_position_weight(position, goals_last_season):
    # Adjust weight based on player position and goal-scoring record
    if position == 'FW':
        return 1.6  # Forwards get the highest weight
    elif position == 'MF':
        if goals_last_season > 5:
            return 1.3  # Attacking midfielders who score often
        else:
            return 0.85  # Defensive/neutral midfielders get a much lower weight
    elif position == 'DF':
        return 0.5  # Defenders get a reduced weight
    elif position == 'GK':
        return 0.2  # Goalkeepers get the lowest weight
    return 1.0  # Default to neutral if position is unknown

# Set the threshold for multiple goals (adjust this value based on testing)
IMPACT_THRESHOLD = 1.5  # Players with a score well above this can score multiple goals
SECOND_GOAL_CHANCE_DECAY = 0.5  # Probability decay for scoring a second goal

# Function to aggregate player data and calculate final scores, with position-based penalty
def aggregate_player_scores(shooting, stats, passing, goals):
    if not shooting.empty:
        players = shooting[['Player', 'goal_prob_scaled', 'Goals']].copy()  # Include goal data
        if not stats.empty:
            players['scoring_impact_scaled'] = stats['scoring_impact_scaled'].values
            players['Match_Play_scaled'] = stats['Match_Play_scaled'].values
            players['90s_played_scaled'] = stats['90s_played_scaled'].values
            players['Pos'] = stats['Pos'].values  # Add position information
        if not passing.empty:
            players['passing_impact_scaled'] = passing['passing_impact_scaled'].values

        # Assign position-based weight, include goals from last season as a factor
        players['position_weight'] = players.apply(lambda row: assign_position_weight(row['Pos'], row['Goals']), axis=1)

        # Calculate final score, including position weight and adjusted for goals
        players['final_score'] = ((players['goal_prob_scaled'] +
                                  players.get('scoring_impact_scaled', 0) +
                                  players.get('passing_impact_scaled', 0) +
                                  0.25 * players.get('Match_Play_scaled', 0) +  # Lesser weight
                                  0.25 * players.get('90s_played_scaled', 0)) / 3.2) * players['position_weight']

        # Sort players by their final score
        sorted_players = players.sort_values(by='final_score', ascending=False)

        # Assign goals to players
        scorers = []
        for _ in range(goals):
            if len(sorted_players) == 0:
                break
            top_scorer = sorted_players.iloc[0]

            # Avoid multiple goals for players with low final score or wrong position
            if top_scorer['final_score'] < 0.7:  # Introduce a cutoff for midfielders/defenders
                continue

            scorers.append(top_scorer['Player'])

            # Reduce probability of multiple goals unless the score is very high
            if top_scorer['final_score'] > IMPACT_THRESHOLD and len(scorers) < goals:
                if random.random() < SECOND_GOAL_CHANCE_DECAY:  # Use probability decay for multiple goals
                    scorers.append(top_scorer['Player'])  # Assign multiple goals to the same player

            # Remove the player if they've scored the assigned number of goals, otherwise re-sort
            sorted_players = sorted_players[1:]

        # Enforce spread rule to ensure variety in scorers if possible
        if len(scorers) > 1:
            unique_scorers = set(scorers)
            if len(unique_scorers) < len(scorers):  # If we have duplicates
                candidates_for_second_goal = [player for player in sorted_players['Player'] if player not in scorers]
                if candidates_for_second_goal:  # Replace the duplicate with a new scorer
                    scorer_to_replace = scorers[-1]  # Last one who scored multiple times
                    replacement = random.choice(candidates_for_second_goal)
                    scorers[-1] = replacement

        return scorers[:goals]  # Return the required number of scorers
    return ["Unknown Scorer"] * goals


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

    # Include 'matches_played' and '90s_played' in stats data, after imputing
    for df in [home_stats, away_stats]:
        if not df.empty:
            for col in ['Match_Play', '90s_played']:
                df[col] = imputer.fit_transform(df[[col]])

    # Normalize features
    scaler = StandardScaler()
    if not home_shooting.empty:
        home_shooting = scale_features(home_shooting, 'goal_prob', scaler)
    if not away_shooting.empty:
        away_shooting = scale_features(away_shooting, 'goal_prob', scaler)

    if not home_stats.empty:
        home_stats = scale_features(home_stats, 'scoring_impact', scaler)
        home_stats = scale_features(home_stats, 'Match_Play', scaler)
        home_stats = scale_features(home_stats, '90s_played', scaler)

    if not away_stats.empty:
        away_stats = scale_features(away_stats, 'scoring_impact', scaler)
        away_stats = scale_features(away_stats, 'Match_Play', scaler)
        away_stats = scale_features(away_stats, '90s_played', scaler)

    if not home_passing.empty:
        home_passing = scale_features(home_passing, 'passing_impact', scaler)
    if not away_passing.empty:
        away_passing = scale_features(away_passing, 'passing_impact', scaler)

    # Set the threshold for multiple goals (adjust this value based on testing)
    IMPACT_THRESHOLD = 1.2  # Players with a score above this can score multiple goals

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

