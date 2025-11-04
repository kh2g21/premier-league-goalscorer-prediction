import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
random_seed = 42
np.random.seed(random_seed)

# -----------------------------
# Load datasets
# -----------------------------
matchday_data = pd.read_csv('2023_matchday_results.csv')
shooting_data = pd.read_csv('player_premier_league_shooting.csv')
stats_data = pd.read_csv('player_premier_league_stats.csv')
passing_data = pd.read_csv('player_premier_league_passing.csv')

# -----------------------------
# Feature Engineering
# -----------------------------
def calculate_features(df, feature_type):
    df = df.copy()
    if feature_type == 'shooting':
        df['goal_per_shot'] = df['Goals'] / (df['Tot_Shot'] + 1)
        df['goal_prob'] = 0.8 * df['Goals'] + 0.45 * df['xG']
    elif feature_type == 'stats':
        df['scoring_impact'] = 0.8 * df['Goals'] + 0.6 * df['xG'] + 0.6 * df['Assist']
    elif feature_type == 'passing':
        df['passing_impact'] = df['KeyPas'] + df['ProgPass']
    return df

def merge_player_data(shooting, stats, passing):
    df = shooting[['Player', 'Squad', 'Pos', 'Goals', 'Tot_Shot', 'xG']].copy()
    df = df.merge(stats[['Player', 'Match_Play', '90s_played', 'Assist']], on='Player', how='left')
    df = df.merge(passing[['Player', 'KeyPas', 'ProgPass']], on='Player', how='left')

    # Feature engineering
    df = calculate_features(df, 'shooting')
    df = calculate_features(df, 'stats')
    df = calculate_features(df, 'passing')

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Scale features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

player_data = merge_player_data(shooting_data, stats_data, passing_data)

# -----------------------------
# Prepare target variable
# -----------------------------
def prepare_targets(match_data, player_df):
    rows = []
    for idx, match in match_data.iterrows():
        home_team = match['teams.home.name']
        away_team = match['teams.away.name']
        home_goals = int(match['goals.home'])
        away_goals = int(match['goals.away'])

        home_players = player_df[player_df['Squad'] == home_team]
        away_players = player_df[player_df['Squad'] == away_team]

        if home_goals > 0:
            top_home_scorers = home_players.sort_values('goal_prob', ascending=False).head(home_goals)
            for player in home_players['Player']:
                target = 1 if player in top_home_scorers['Player'].values else 0
                rows.append({'Player': player, 'Squad': home_team, 'target': target})
        else:
            for player in home_players['Player']:
                rows.append({'Player': player, 'Squad': home_team, 'target': 0})

        if away_goals > 0:
            top_away_scorers = away_players.sort_values('goal_prob', ascending=False).head(away_goals)
            for player in away_players['Player']:
                target = 1 if player in top_away_scorers['Player'].values else 0
                rows.append({'Player': player, 'Squad': away_team, 'target': target})
        else:
            for player in away_players['Player']:
                rows.append({'Player': player, 'Squad': away_team, 'target': 0})

    target_df = pd.DataFrame(rows)
    return target_df

target_df = prepare_targets(matchday_data, player_data)
ml_data = player_data.merge(target_df[['Player', 'target']], on='Player', how='left').fillna(0)

# -----------------------------
# Features and labels
# -----------------------------
feature_cols = ['goal_per_shot', 'goal_prob', 'scoring_impact', 'passing_impact',
                'Match_Play', '90s_played', 'Tot_Shot', 'xG', 'Assist', 'KeyPas', 'ProgPass']

X = ml_data[feature_cols]
y = ml_data['target']

# -----------------------------
# Memory-safe Hyperparameter Tuning
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

models = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=random_seed),
        "params": {
            'n_estimators': [100, 150, 200],
            'max_depth': [5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=random_seed),
        "params": {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=random_seed),
        "params": {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
    }
}

best_models = {}
best_scores = {}

for name, mp in models.items():
    print(f"Tuning {name} ...")
    grid = RandomizedSearchCV(
        mp['model'],
        mp['params'],
        cv=kf,
        scoring='roc_auc',
        n_iter=10,
        n_jobs=1,  # memory-safe
        random_state=random_seed
    )
    grid.fit(X, y)
    best_models[name] = grid.best_estimator_
    best_scores[name] = grid.best_score_
    print(f"{name} Best AUC: {grid.best_score_:.4f}\n")

# -----------------------------
# Select the overall best model
# -----------------------------
best_model_name = max(best_scores, key=best_scores.get)
final_model = best_models[best_model_name]
print(f"Selected Model: {best_model_name} with AUC {best_scores[best_model_name]:.4f}")

# -----------------------------
# Feature Importance (tree-based)
# -----------------------------
def plot_feature_importance(model, feature_names, title="Feature Importance"):
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10,6))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title(title)
        plt.show()

if best_model_name in ['RandomForest', 'GradientBoosting']:
    plot_feature_importance(final_model, feature_cols, title=f"{best_model_name} Feature Importance")

# -----------------------------
# Predict top scorers for a match
# -----------------------------
def predict_match_scorers(home_team, away_team, home_goals, away_goals, player_df, model):
    home_players = player_df[player_df['Squad'] == home_team].copy()
    away_players = player_df[player_df['Squad'] == away_team].copy()

    home_players['score_prob'] = model.predict_proba(home_players[feature_cols])[:,1]
    away_players['score_prob'] = model.predict_proba(away_players[feature_cols])[:,1]

    top_home_scorers = home_players.sort_values('score_prob', ascending=False).head(home_goals)['Player'].tolist()
    top_away_scorers = away_players.sort_values('score_prob', ascending=False).head(away_goals)['Player'].tolist()

    return top_home_scorers, top_away_scorers

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

# -----------------------------
# Predict scorers for all matches
# -----------------------------
for idx, row in matchday_data.iterrows():
    home_team = row['teams.home.name']
    away_team = row['teams.away.name']
    home_goals = int(row['goals.home'])
    away_goals = int(row['goals.away'])

    if home_goals > 0 or away_goals > 0:
        home_scorers, away_scorers = predict_match_scorers(home_team, away_team, home_goals, away_goals, player_data, final_model)
    else:
        home_scorers, away_scorers = [], []

    print(f"Match: {home_team} vs {away_team}")
    print(f"Home Goals: {home_goals} - Scorers: {', '.join(home_scorers) if home_scorers else 'None'}")
    print(f"Away Goals: {away_goals} - Scorers: {', '.join(away_scorers) if away_scorers else 'None'}\n")
