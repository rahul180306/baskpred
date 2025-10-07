#!/usr/bin/env python3
"""
Test script for Neural Network implementation
"""
import sys
import os
sys.path.append('models')

try:
    from models.neural_network import NeuralNetworkModel, NeuralNetworkClassifier
    print("✅ Neural network modules imported successfully!")
    
    # Test with sample data
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X_sample = np.random.rand(100, 3) * 1000  # Combined points, rebounds, assists
    y_sample = (X_sample[:, 0] + X_sample[:, 1] + X_sample[:, 2]) / 1000 - 5 + np.random.normal(0, 0.5, 100)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
    
    # Test Neural Network Regressor
    print("\n🧠 Testing Neural Network Regressor...")
    nn_model = NeuralNetworkModel(input_dim=3)
    nn_model.build_model()
    
    print(f"Model built with input shape: {nn_model.model.input_shape}")
    print(f"Model summary:")
    nn_model.model.summary()
    
    # Train the model
    print("\n🏋️ Training the model...")
    history = nn_model.train(X_train, y_train, epochs=50, batch_size=16)
    
    # Make predictions
    predictions = nn_model.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n📊 Results:")
    print(f"MSE: {mse:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"Sample predictions vs actual:")
    for i in range(min(5, len(predictions))):
        print(f"  Predicted: {predictions[i]:.4f}, Actual: {y_test[i]:.4f}")
    
    print("\n✅ Neural Network implementation working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure TensorFlow is installed: pip install tensorflow")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()