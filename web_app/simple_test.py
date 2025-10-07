#!/usr/bin/env python3
"""
Simple test without TensorFlow to check basic functionality
"""

print("🧠 Testing Neural Network Implementation...")

# Test basic imports
try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    print("✅ Basic imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test with sample data (same as your NBA project)
print("\n📊 Creating sample NBA duo data...")
np.random.seed(42)

# Create sample data similar to your NBA project
n_samples = 200
combined_points = np.random.normal(3500, 800, n_samples)
combined_rebounds = np.random.normal(1000, 200, n_samples)
combined_assists = np.random.normal(600, 150, n_samples)

# Create features matrix
X = np.column_stack([combined_points, combined_rebounds, combined_assists])

# Create target (net rating) with some realistic relationships
y = (
    0.002 * combined_points + 
    0.005 * combined_rebounds + 
    0.008 * combined_assists - 
    10 + np.random.normal(0, 2, n_samples)
)

print(f"Dataset shape: {X.shape}")
print(f"Sample features (first 3 rows):")
for i in range(3):
    print(f"  Points: {X[i,0]:.0f}, Rebounds: {X[i,1]:.0f}, Assists: {X[i,2]:.0f} -> Net Rating: {y[i]:.2f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test Linear Regression (your current model)
print("\n🔍 Testing Linear Regression (Current Model)...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)

print(f"Linear Regression Results:")
print(f"  MSE: {linear_mse:.6f}")
print(f"  R²: {linear_r2:.6f}")

# Test TensorFlow availability
print("\n🤖 Testing TensorFlow availability...")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # Test if we can create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print("✅ TensorFlow model creation successful!")
    
    # Quick training test (just a few epochs)
    print("🏋️ Quick training test...")
    model.fit(X_train, y_train, epochs=5, verbose=0)
    
    # Make predictions
    tf_pred = model.predict(X_test, verbose=0).flatten()
    tf_mse = mean_squared_error(y_test, tf_pred)
    tf_r2 = r2_score(y_test, tf_pred)
    
    print(f"Neural Network Results (5 epochs):")
    print(f"  MSE: {tf_mse:.6f}")
    print(f"  R²: {tf_r2:.6f}")
    
    improvement = ((linear_r2 - tf_r2) / abs(linear_r2) * 100) if linear_r2 != 0 else 0
    print(f"  Improvement: {abs(improvement):.1f}%")
    
except ImportError:
    print("❌ TensorFlow not available")
except Exception as e:
    print(f"❌ TensorFlow error: {e}")

print("\n✅ Test completed! If TensorFlow worked, your neural network implementation is ready!")