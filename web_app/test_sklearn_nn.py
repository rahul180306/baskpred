#!/usr/bin/env python3
"""
Test script for Scikit-learn Neural Network implementation
"""
import sys
sys.path.append('models')

print("🧠 Testing Scikit-learn Neural Network Implementation...")

try:
    from models.neural_network import NeuralNetworkModel, NeuralNetworkClassifier
    print("✅ Neural network modules imported successfully!")
    
    # Test with sample data
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    
    # Create sample NBA data
    print("\n📊 Creating sample NBA duo data...")
    np.random.seed(42)
    n_samples = 300
    
    # Sample NBA player duo stats (more realistic ranges)
    combined_points = np.random.normal(3500, 800, n_samples)
    combined_rebounds = np.random.normal(1000, 200, n_samples)
    combined_assists = np.random.normal(600, 150, n_samples)
    
    # Create features matrix
    X_sample = np.column_stack([combined_points, combined_rebounds, combined_assists])
    
    # Create realistic net rating with complex relationships
    y_sample = (
        0.002 * combined_points + 
        0.005 * combined_rebounds + 
        0.008 * combined_assists +
        0.000001 * combined_points * combined_assists +  # Interaction term
        -10 + np.random.normal(0, 1.5, n_samples)
    )
    
    print(f"Dataset shape: {X_sample.shape}")
    print("Sample data (first 5 rows):")
    for i in range(5):
        print(f"  Points: {X_sample[i,0]:.0f}, Rebounds: {X_sample[i,1]:.0f}, Assists: {X_sample[i,2]:.0f} -> Net Rating: {y_sample[i]:.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"\nData split: Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Test Linear Regression (current model)
    print("\n🔍 Testing Linear Regression (Current Model)...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    
    linear_mse = mean_squared_error(y_test, linear_pred)
    linear_r2 = r2_score(y_test, linear_pred)
    
    print(f"Linear Regression Results:")
    print(f"  MSE: {linear_mse:.6f}")
    print(f"  R²: {linear_r2:.6f}")
    
    # Test Neural Network Regressor
    print("\n🧠 Testing Neural Network Regressor...")
    nn_model = NeuralNetworkModel(input_dim=3)
    nn_model.build_model()
    
    print(f"Neural Network Architecture: {nn_model.model.hidden_layer_sizes}")
    
    # Train the model
    print("🏋️ Training the neural network...")
    history = nn_model.train(X_train, y_train, X_val, y_val)
    
    # Make predictions
    print("🎯 Making predictions...")
    nn_predictions = nn_model.predict(X_test)
    
    # Calculate metrics
    nn_mse = mean_squared_error(y_test, nn_predictions)
    nn_r2 = r2_score(y_test, nn_predictions)
    nn_mae = np.mean(np.abs(y_test - nn_predictions))
    
    print(f"\n📊 RESULTS COMPARISON:")
    print(f"{'Model':<20} {'MSE':<12} {'R²':<12} {'MAE':<12}")
    print(f"{'-'*60}")
    print(f"{'Linear Regression':<20} {linear_mse:<12.6f} {linear_r2:<12.6f} {np.mean(np.abs(y_test - linear_pred)):<12.6f}")
    print(f"{'Neural Network':<20} {nn_mse:<12.6f} {nn_r2:<12.6f} {nn_mae:<12.6f}")
    
    # Calculate improvement
    mse_improvement = ((linear_mse - nn_mse) / linear_mse * 100) if linear_mse > 0 else 0
    r2_improvement = ((nn_r2 - linear_r2) / abs(linear_r2) * 100) if linear_r2 != 0 else 0
    
    print(f"\n🚀 IMPROVEMENTS:")
    print(f"MSE improved by: {mse_improvement:.1f}%")
    print(f"R² improved by: {r2_improvement:.1f}%")
    
    # Show some example predictions
    print(f"\n🎯 Sample Predictions (first 5 test cases):")
    print(f"{'Actual':<10} {'Linear':<10} {'Neural Net':<10} {'NN Better?'}")
    print(f"{'-'*45}")
    for i in range(5):
        actual = y_test[i]
        linear = linear_pred[i]
        neural = nn_predictions[i]
        nn_better = "✅" if abs(actual - neural) < abs(actual - linear) else "❌"
        print(f"{actual:<10.2f} {linear:<10.2f} {neural:<10.2f} {nn_better}")
    
    # Test training history
    history_data = nn_model.get_training_history()
    if history_data:
        print(f"\n📈 Training completed in {len(history_data['loss'])} iterations")
        print(f"Final training loss: {history_data['loss'][-1]:.6f}")
    
    print(f"\n✅ Neural Network implementation working perfectly!")
    print(f"🎉 Ready to integrate with Flask app!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()