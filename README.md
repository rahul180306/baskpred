<!-- cspell:ignore kagglehub numpy scikit statsmodels ipynb Nemar's Kaggle -->

# NBA Player Synergy Prediction

This project builds machine learning models to predict synergy between NBA players using play-by-play data from 1997-2023.

## Features

- **Regression Model**: Multiple Linear Regression to predict Net Rating for player duos based on combined individual stats.
- **Classification Models**: Logistic Regression and Decision Tree to classify synergy levels (High, Neutral, Low).
- **Evaluation**: MSE, R-squared, Accuracy, Cross-Validation with KFold.
- **Model Comparison**: McNemar's Test to compare classification models.

## Data

- Source: Kaggle NBA Play-by-Play Data (1997-2023)
- Processed to extract player stats (points, rebounds, assists) and synthetic duo net ratings.

## Requirements

- Python 3.13
- `kagglehub`
- pandas
- numpy
- scikit-learn
- `statsmodels`

Install with: `pip install -r requirements.txt`

## Usage

Run the Jupyter notebook `nba_synergy.ipynb` to train and evaluate the models.

## Results

- Regression: Perfect fit on synthetic data (MSE ~0, R²=1.0)
- Classification: Logistic Regression outperforms Decision Tree (73.7% vs 60.5% accuracy)
- McNemar's Test: No significant difference between classifiers (p=0.18)
# baskpred
