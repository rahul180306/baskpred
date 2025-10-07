from flask import (Flask, render_template, request, jsonify)
import pandas as pd
import os
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     KFold)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from statsmodels.stats.contingency_tables import (
    mcnemar  # spell-checker: ignore
)
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Import neural network models
try:
    from models.neural_network import (
        NeuralNetworkModel, NeuralNetworkClassifier)
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError as e:
    print(f"Neural networks not available: {e}")
    NEURAL_NETWORKS_AVAILABLE = False

app = Flask(__name__)

# Constants
PLEASE_LOAD_DATA_FIRST = "Please load data first."
PLEASE_PROCESS_DATA_FIRST = "Please process data first."
ADVANCED_MODELS_TEMPLATE = 'advanced_models.html'
DATA_LOADING_TEMPLATE = 'data_loading.html'
DATA_PROCESSING_TEMPLATE = 'data_processing.html'
REGRESSION_TEMPLATE = 'regression.html'
CLASSIFICATION_TEMPLATE = 'classification.html'
COMPARISON_TEMPLATE = 'comparison.html'
REGRESSION_TEMPLATE = 'regression.html'
CLASSIFICATION_TEMPLATE = 'classification.html'
COMPARISON_TEMPLATE = 'comparison.html'

# Global variables to store data and models
data_loaded = False
data_processed = False
models_trained = False
advanced_models_trained = False
duo_df = None
X = None
y = None
y_class = None
X_train = None
X_test = None
y_train = None
y_test = None
y_class_train = None
y_class_test = None
model = None
log_model = None
dt_model = None
log_pred = None
dt_pred = None
X_train_c = None
X_test_c = None
y_train_c = None
y_test_c = None

# Neural network models
nn_regressor = None
nn_classifier = None
nn_results = None
nn_classifier_results = None
training_history = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/data_loading')
def data_loading():
    global data_loaded
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "szymonjwiak/nba-play-by-play-data-1997-2023")
        
        # Get year from query parameter, default to 2022
        year = request.args.get('year', '2022')
        try:
            year = int(year)
        except ValueError:
            year = 2023
        
        # Check if file exists
        file_path = os.path.join(path, f'pbp{year}.csv')
        if not os.path.exists(file_path):
            return render_template(DATA_LOADING_TEMPLATE,
                                   message=f"Data for year {year} not available.",
                                   columns=[],
                                   teams=[],
                                   team_colors={},
                                   players_by_team={},
                                   player_stats={},
                                   selected_year=year)
        
        # Load data for the selected year
        df = pd.read_csv(file_path)
        
        data_loaded = True
        
        # Extract teams
        if 'team' in df.columns:
            teams = sorted(df['team'].dropna().unique())
        else:
            teams = []
            # If no 'team' column, can't group players by team
            return render_template(DATA_LOADING_TEMPLATE,
                                   message=f"Dataset for {year} loaded, but no 'team' column found to group players.",
                                   columns=df.columns.tolist()[:10],
                                   teams=[],
                                   team_colors={},
                                   players_by_team={},
                                   player_stats={},
                                   player_names={},
                                   selected_year=year)
        
        # Assign colors to teams
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        team_colors = {team: colors[i % len(colors)]
                       for i, team in enumerate(teams)}
        
        # Get player names
        if 'player' in df.columns:
            player_names_df = df[['playerid', 'player']].drop_duplicates()
            player_names = {int(row['playerid']): row['player'] for _, row in player_names_df.iterrows()}
        else:
            player_names = {}
        
        # Get players by team
        players_by_team = {}
        player_stats = {}
        for team in teams:
            team_events = df[df['team'] == team]
            players = team_events['playerid'].dropna().unique()
            players_by_team[team] = [int(p) for p in players]
            
            for player in players:
                # Simple stats: count of events
                count = len(df[df['playerid'] == player])
                player_stats[int(player)] = {'events': int(count),
                                             'team': team}
        
        return render_template(DATA_LOADING_TEMPLATE,
                               message=f"Dataset for {year} loaded successfully!",
                               columns=df.columns.tolist()[:10],
                               teams=teams,
                               team_colors=team_colors,
                               players_by_team=players_by_team,
                               player_stats=player_stats,
                               player_names=player_names,
                               selected_year=year)
    except Exception as e:
        return render_template(DATA_LOADING_TEMPLATE,
                               message=f"Error: {str(e)}",
                               columns=[],
                               teams=[],
                               team_colors={},
                               players_by_team={},
                               player_stats={},
                               player_names={},
                               selected_year=year)


@app.route('/data_processing')
def data_processing():
    global data_processed, duo_df, X, y, y_class
    if not data_loaded:
        return render_template(DATA_PROCESSING_TEMPLATE,
                               message=PLEASE_LOAD_DATA_FIRST)
    try:
        path = ('/Users/rahulkarthikt/.cache/kagglehub/datasets/'
                'szymonjwiak/nba-play-by-play-data-1997-2023/versions/1')
        years = [2010, 2011, 2012]
        dfs = []
        for year in years:
            df = pd.read_csv(os.path.join(path, f'pbp{year}.csv'))
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)

        df_shots = df_all[df_all['type'] == 'Made Shot']
        df_shots = df_shots.copy()
        df_shots['pts'] = (df_shots['desc']
                           .str.extract(r'\((\d+) PTS\)')[0]
                           .astype(float))
        player_pts = (df_shots.groupby(['playerid', 'season'])['pts']
                      .sum().reset_index(name='pts'))

        df_rebounds = df_all[df_all['type'] == 'Rebound']
        player_reb = (df_rebounds.groupby(['playerid', 'season'])
                      .size().reset_index(name='reb'))

        df_assists = df_all[df_all['subtype'] == 'Assist']
        player_ast = (df_assists.groupby(['playerid', 'season'])
                      .size().reset_index(name='ast'))

        player_stats = (player_pts
                        .merge(
                            player_reb,
                            on=['playerid', 'season'],
                            how='outer'
                        )
                        .merge(
                            player_ast,
                            on=['playerid', 'season'],
                            how='outer'
                        )
                        .fillna(0))

        top_players = (player_stats.groupby('playerid')['pts']
                       .sum().nlargest(20).index)
        duo_list = []
        for i in range(len(top_players)):
            for j in range(i+1, len(top_players)):
                duo_list.append((top_players[i], top_players[j]))
        duo_df = pd.DataFrame(duo_list, columns=['player_a', 'player_b'])
        duo_df['season'] = 2010

        player_a_stats = player_stats.rename(
            columns={'playerid': 'player_a', 'pts': 'pts_a', 'reb': 'reb_a',
                     'ast': 'ast_a'})
        player_b_stats = player_stats.rename(
            columns={'playerid': 'player_b', 'pts': 'pts_b', 'reb': 'reb_b',
                     'ast': 'ast_b'})

        duo_df = duo_df.merge(
            player_a_stats[['player_a', 'season', 'pts_a', 'reb_a', 'ast_a']],
            on=['player_a', 'season'], how='left', validate='many_to_one')
        duo_df = duo_df.merge(
            player_b_stats[['player_b', 'season', 'pts_b', 'reb_b', 'ast_b']],
            on=['player_b', 'season'], how='left', validate='many_to_one')

        duo_df['combined_pts'] = duo_df['pts_a'] + duo_df['pts_b']
        duo_df['combined_reb'] = duo_df['reb_a'] + duo_df['reb_b']
        duo_df['combined_ast'] = duo_df['ast_a'] + duo_df['ast_b']

        rng = np.random.default_rng(42)
        duo_df['net_rating'] = (
            (duo_df['combined_pts'] + duo_df['combined_reb'] +
             duo_df['combined_ast']) / 10000 - 5 +
            rng.normal(0, 1, len(duo_df)))

        X = duo_df[['combined_pts', 'combined_reb', 'combined_ast']]
        y = duo_df['net_rating']
        data_processed = True
        return render_template(DATA_PROCESSING_TEMPLATE,
                               message="Data processed successfully!",
                               sample=duo_df.head().to_html())
    except Exception as e:
        return render_template(DATA_PROCESSING_TEMPLATE,
                               message=f"Error: {str(e)}")


@app.route('/regression')
def regression():
    global models_trained, X_train, X_test, y_train, y_test, model
    if not data_processed:
        return render_template(REGRESSION_TEMPLATE,
                               message=PLEASE_PROCESS_DATA_FIRST)
    try:
        (X_train, X_test, y_train, y_test) = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        models_trained = True
        return render_template(REGRESSION_TEMPLATE,
                               message="Regression model trained.",
                               mse=mse, r2=r2)
    except Exception as e:
        return render_template(REGRESSION_TEMPLATE,
                               message=f"Error: {str(e)}")


@app.route('/classification')
def classification():
    global y_class, X_train_c, X_test_c, y_train_c, y_test_c, \
           log_model, dt_model, log_pred, dt_pred
    if not data_processed:
        return render_template(CLASSIFICATION_TEMPLATE,
                               message=PLEASE_PROCESS_DATA_FIRST)
    try:
        duo_df['synergy_class'] = pd.cut(  # type: ignore
            duo_df['net_rating'],  # type: ignore
            bins=[-float('inf'), -3.5, 3.5, float('inf')],
            labels=[0, 1, 2])
        y_class = duo_df['synergy_class']  # type: ignore
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X, y_class, test_size=0.2, random_state=42)

        log_model = LogisticRegression(random_state=42)
        log_model.fit(X_train_c, y_train_c)

        dt_model = DecisionTreeClassifier(ccp_alpha=0.0, random_state=42)
        dt_model.fit(X_train_c, y_train_c)

        log_pred = log_model.predict(X_test_c)
        dt_pred = dt_model.predict(X_test_c)

        log_acc = accuracy_score(y_test_c, log_pred)
        dt_acc = accuracy_score(y_test_c, dt_pred)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        log_cv = cross_val_score(log_model, X, y_class, cv=kf)  # type: ignore
        dt_cv = cross_val_score(dt_model, X, y_class, cv=kf)  # type: ignore

        return render_template(CLASSIFICATION_TEMPLATE,
                               message="Classification models trained.",
                               log_acc=log_acc, dt_acc=dt_acc,
                               log_cv=log_cv.mean(), dt_cv=dt_cv.mean())
    except Exception as e:
        return render_template(CLASSIFICATION_TEMPLATE,
                               message=f"Error: {str(e)}")


@app.route('/comparison')
def comparison():
    if log_pred is None or dt_pred is None:
        return render_template(
            COMPARISON_TEMPLATE,
            message="Please train classification models first.")
    try:
        table = [[0, 0], [0, 0]]
        for i in range(len(y_test_c)):  # type: ignore
            if (y_test_c.iloc[i] == dt_pred[i] and  # type: ignore
                    y_test_c.iloc[i] != log_pred[i]):  # type: ignore
                table[0][1] += 1
            elif (y_test_c.iloc[i] != dt_pred[i] and  # type: ignore
                  y_test_c.iloc[i] == log_pred[i]):  # type: ignore
                table[1][0] += 1

        for i in range(len(y_test_c)):  # type: ignore
            if (y_test_c.iloc[i] == dt_pred[i] and  # type: ignore
                    y_test_c.iloc[i] == log_pred[i]):  # type: ignore
                table[0][0] += 1
            elif (y_test_c.iloc[i] != dt_pred[i] and  # type: ignore
                  y_test_c.iloc[i] != log_pred[i]):  # type: ignore
                table[1][1] += 1

        result = mcnemar(table, exact=True)
        stat = getattr(result, 'statistic')
        pval = getattr(result, 'pvalue')

        return render_template(COMPARISON_TEMPLATE,
                               message="Models compared.",
                               table=table, stat=stat, pval=pval)
    except Exception as e:
        return render_template(COMPARISON_TEMPLATE,
                               message=f"Error: {str(e)}")


@app.route('/advanced_models')
def advanced_models():
    global nn_regressor, nn_classifier, nn_results, \
           nn_classifier_results, training_history, advanced_models_trained
    
    if not data_processed:
        return render_template(ADVANCED_MODELS_TEMPLATE,
                               message=PLEASE_PROCESS_DATA_FIRST)
    
    if not NEURAL_NETWORKS_AVAILABLE:
        return render_template(
            ADVANCED_MODELS_TEMPLATE,
            error=("Neural networks not available. Please install TensorFlow: "
                   "pip install tensorflow"))
    
    try:
        # Split data with validation set for neural networks
        x_train_nn, x_temp, y_train_nn, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42)
        x_val, x_test_nn, y_val, y_test_nn = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=42)
        
        # Train Neural Network Regressor
        nn_regressor = NeuralNetworkModel(input_dim=X.shape[1])
        nn_regressor.build_model()
        nn_history = nn_regressor.train(x_train_nn, y_train_nn,
                                        x_val, y_val, epochs=150)
        
        # Evaluate neural network regressor
        nn_pred = nn_regressor.predict(x_test_nn)
        nn_mse = mean_squared_error(y_test_nn, nn_pred)
        nn_r2 = r2_score(y_test_nn, nn_pred)
        nn_mae = np.mean(np.abs(y_test_nn - nn_pred))
        
        # Get training history for plotting
        training_history = json.dumps(nn_regressor.get_training_history())
        
        # Store results
        nn_results = {
            'mse': nn_mse,
            'r2': nn_r2,
            'mae': nn_mae,
            'epochs_trained': len(nn_history.history['loss'])
        }
        
        # Get linear regression results for comparison
        if model is not None:
            linear_pred = model.predict(x_test_nn)
            linear_mse = mean_squared_error(y_test_nn, linear_pred)
            linear_r2 = r2_score(y_test_nn, linear_pred)
        else:
            # Train a quick linear model for comparison
            temp_model = LinearRegression()
            temp_model.fit(x_train_nn, y_train_nn)
            linear_pred = temp_model.predict(x_test_nn)
            linear_mse = mean_squared_error(y_test_nn, linear_pred)
            linear_r2 = r2_score(y_test_nn, linear_pred)
            
        linear_results = {
            'mse': linear_mse,
            'r2': linear_r2
        }
        
        # Train Neural Network Classifier if classification data is available
        nn_classifier_results = None
        classification_results = None
        
        if y_class is not None:
            # Prepare classification data
            y_class_encoded = pd.Categorical(y_class).codes
            x_train_c_nn, x_temp_c, y_train_c_nn, y_temp_c = train_test_split(
                X, y_class_encoded, test_size=0.4, random_state=42)
            x_val_c, x_test_c_nn, y_val_c, y_test_c_nn = train_test_split(
                x_temp_c, y_temp_c, test_size=0.5, random_state=42)
            
            # Train neural network classifier
            nn_classifier = NeuralNetworkClassifier(
                input_dim=X.shape[1],
                num_classes=len(np.unique(y_class_encoded)))
            nn_classifier.build_model()
            nn_classifier.train(x_train_c_nn, y_train_c_nn,
                                x_val_c, y_val_c, epochs=100)
            
            # Evaluate neural network classifier
            nn_class_pred = nn_classifier.predict(x_test_c_nn)
            nn_accuracy = accuracy_score(y_test_c_nn, nn_class_pred)
            
            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            nn_precision = precision_score(y_test_c_nn, nn_class_pred,
                                           average='weighted')
            nn_recall = recall_score(y_test_c_nn, nn_class_pred,
                                     average='weighted')
            nn_f1 = f1_score(y_test_c_nn, nn_class_pred,
                             average='weighted')
            
            nn_classifier_results = {
                'accuracy': nn_accuracy,
                'precision': nn_precision,
                'recall': nn_recall,
                'f1': nn_f1
            }
            
            # Get traditional classification results for comparison
            if log_model is not None and dt_model is not None:
                log_pred_temp = log_model.predict(x_test_c_nn)
                dt_pred_temp = dt_model.predict(x_test_c_nn)
                log_acc = accuracy_score(y_test_c_nn, log_pred_temp)
                dt_acc = accuracy_score(y_test_c_nn, dt_pred_temp)
            else:
                # Train quick models for comparison
                temp_log = LogisticRegression(random_state=42, max_iter=1000)
                temp_dt = DecisionTreeClassifier(ccp_alpha=0.0,
                                                 random_state=42)
                temp_log.fit(x_train_c_nn, y_train_c_nn)
                temp_dt.fit(x_train_c_nn, y_train_c_nn)
                log_pred_temp = temp_log.predict(x_test_c_nn)
                dt_pred_temp = temp_dt.predict(x_test_c_nn)
                log_acc = accuracy_score(y_test_c_nn, log_pred_temp)
                dt_acc = accuracy_score(y_test_c_nn, dt_pred_temp)
                
            classification_results = {
                'log_acc': log_acc,
                'dt_acc': dt_acc
            }
        
        advanced_models_trained = True
        
        return render_template(ADVANCED_MODELS_TEMPLATE,
                               message="Advanced models trained successfully!",
                               nn_results=nn_results,
                               linear_results=linear_results,
                               nn_classifier_results=nn_classifier_results,
                               classification_results=classification_results,
                               training_history=training_history)
                               
    except Exception as e:
        return render_template(ADVANCED_MODELS_TEMPLATE,
                               error=f"Error training models: {str(e)}")


@app.route('/predict_synergy', methods=['POST'])
def predict_synergy():
    """Real-time prediction endpoint for all trained models"""
    try:
        # Get player stats from request
        data = request.json
        if data is None:
            return jsonify({'error': 'No data provided'})
        player_stats = np.array([[
            float(data['combined_pts']),
            float(data['combined_reb']),
            float(data['combined_ast'])
        ]])
        
        predictions = {}
        
        # Get predictions from all available models
        if model is not None:
            predictions['linear_regression'] = float(
                model.predict(player_stats)[0])
        
        if nn_regressor is not None:
            predictions['neural_network'] = float(
                nn_regressor.predict(player_stats)[0])
            
        # If no models are trained, return error
        if not predictions:
            return jsonify({
                'error': ('No models have been trained yet. '
                          'Please train models first.')})
            
        return jsonify({'predictions': predictions})
        
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
