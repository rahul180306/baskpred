#!/usr/bin/env python3
"""
Simple Flask app starter that shows current status
"""

print("🚀 Starting NBA Synergy Prediction Flask App...")

try:
    # Import the app
    from app import app
    print("✅ Flask app imported successfully!")
    
    # Check neural network availability
    from app import NEURAL_NETWORKS_AVAILABLE
    if NEURAL_NETWORKS_AVAILABLE:
        print("✅ Neural networks available!")
    else:
        print("⚠️ Neural networks not available")
    
    # Show available routes
    print("\n📍 Available Routes:")
    print("- Home: http://127.0.0.1:5000/")
    print("- Data Loading: http://127.0.0.1:5000/data_loading")
    print("- Data Processing: http://127.0.0.1:5000/data_processing")
    print("- Regression: http://127.0.0.1:5000/regression")
    print("- Classification: http://127.0.0.1:5000/classification")
    print("- Model Comparison: http://127.0.0.1:5000/comparison")
    print("- Advanced Models: http://127.0.0.1:5000/advanced_models")
    
    print("\n🎯 Workflow:")
    print("1. Start with Data Loading")
    print("2. Then Data Processing")
    print("3. Then Regression (trains linear model)")
    print("4. Then Classification (optional)")
    print("5. Finally Advanced Models (neural networks)")
    
    print("\n🌐 Starting Flask server...")
    app.run(debug=True, host='127.0.0.1', port=5000)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()