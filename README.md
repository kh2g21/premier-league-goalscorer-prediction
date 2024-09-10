# Football Goalscorer Prediction

## Description

This project predicts football goalscorers for a given match using player performance data. It combines several independent datasets, such as shooting, passing, and overall player statistics, to assess which players are most likely to score in a match. The model processes historical match results and player stats, applies feature engineering, and ranks players based on computed features, such as goal probability, scoring impact, and passing impact. The aim is to provide accurate predictions of who might score in upcoming matches for both home and away teams.

This approach is rooted in the principle that:
- Players with better historical shooting accuracy, higher expected goals (xG), and active involvement in the attacking play should have a higher probability of scoring.
- Position on the field impacts goal likelihood: forwards are expected to score more frequently than midfielders and defenders.
- Scoring consistency in real football matches is unpredictable, and our model introduces randomness to mimic this aspect.
  
## Techniques Used

### Feature Engineering
- **Goal Probability**: Calculated by combining expected goals (xG) and goals per shot to reflect both the quantity and quality of shooting opportunities.
- **Scoring Impact**: Incorporates goals, expected goals (xG), and assists from player statistics, offering a broader measure of offensive contribution.
- **Passing Impact**: Derived from key passes and progressive passes, which are indicators of a player's involvement in creating scoring chances.

These features are chosen to capture various aspects of a player's performance: shooting quality, direct contribution to goals, and playmaking ability.

### Imputation for Missing Data
- **Mean Imputation**: Applied to missing numeric values to ensure the model can handle incomplete datasets without losing valuable player information. This ensures that predictions remain robust, even when some player stats are missing.

### Data Normalization
- **StandardScaler**: Used to normalize the computed features across players, so different types of contributions (shooting, scoring, passing) can be meaningfully compared.

### Position-Based Weighting
Players' scoring potential is adjusted based on their positions (e.g., forward, midfielder, defender) and their historical goal-scoring records. Forwards have the highest probability, midfielders have a reduced likelihood (based on goals in prior seasons), and defenders rarely score.

### Aggregation of Player Impact
Final scores are calculated by aggregating various features, such as shooting, passing, and positional impact. These scores represent each player's likelihood of scoring, considering both their attacking stats and contribution to play.

### Goal Distribution Logic
A custom algorithm is used to distribute goals among players based on their final scores, introducing randomness to reflect real-life uncertainty. The model ensures goals are spread across players and only assigns multiple goals to a player if their final score is significantly higher than the rest.

## Datasets Used

All datasets were sourced from Kaggle. Note that all datasets will need to be in same directory as the Python script to execute successfully.

The [2023_matchday_results.csv dataset](https://www.kaggle.com/datasets/afnanurrahim/premier-league-2022-23)

The [player_premier_league_shooting.csv, player_premier_league_stats.csv, player_premier_league_passing.csv](https://www.kaggle.com/datasets/mechatronixs/english-premier-league-22-23-season-stats) datasets

- **`2023_matchday_results.csv `**: Contains the home and away teams and the actual goals scored in each match.
- **`player_premier_league_shooting.csv`**: Includes player-level shooting metrics such as total shots, goals, and expected goals (xG).
- **`player_premier_league_stats.csv`**: Provides detailed player statistics, including goals, assists, and expected goals.
- **`player_premier_league_passing.csv`**: Includes passing metrics such as key passes and progressive passes, which contribute to goal creation.

## How the Project Could Be Improved

1. **Integrating Additional Features**:
   - **Defensive Metrics**: Using a defensive stats (e.g., tackles, interceptions) dataset from the same season could help better assess a player's overall influence on matches, especially for midfielders. However, I chose not to integrate this into my project, because I did not want the model to be too influenced or impacted by defensive data; as this information would largely be inconsequential when assessing likely goalscorers. 
   - **Form and Fitness Data**: Having access to data about recent form and player fitness could further enhance prediction accuracy.
   - **Match Context**: External factors such as home advantage, and fixture congestion could add valuable context to predictions.

2. **Advanced Machine Learning Models**:
   - Implementing more sophisticated algorithms such as Gradient Boosting, Neural Networks, or ensemble methods could improve predictive accuracy by capturing more complex patterns in the data. For example, looking at trends in performance in previous games to predict which players are likely to score, or performance against opponent in reverse fixture earlier in the season may aid accuracy of prediction. 

3. **Team-Level Data**:
   - Incorporating team-level data (e.g., team possession, pressing intensity) could complement individual player stats and provide a more holistic view of match dynamics.

## Limitations

1. **Limited Feature Set**: The current model only considers shooting, passing, and basic player statistics. It does not account for external factors like team strategies, defensive setups, player confidence/morale or match-specific conditions which are difficult to quantify.
   
2. **Static Nature**: The model doesn't incorporate real-time in-game events (e.g., substitutions, injuries, red cards) that could drastically change goal-scoring probabilities.

3. **Bias in Data Used**: Players with limited playtime or involvement may not have sufficient data points, which can lead to under-representation in predictions. Furthermore, the datasets used are limited to the 2022-23 season only; therefore underlying historical trends will be missed (e.g player's historical performance against opponent)

By addressing these limitations, the model could provide even more accurate and insightful predictions for football match outcomes.
