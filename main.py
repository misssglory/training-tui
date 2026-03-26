#!/usr/bin/env python
"""
Transformer TUI Application
Main entry point for the Textual-based transformer training and testing interface.
"""

import sys
import os
import json
from pathlib import Path

# Handle TensorFlow import based on configuration
def setup_tensorflow():
    """Configure TensorFlow import based on system settings."""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        tf_source = config.get("system", {}).get("tensorflow_source", "pip")
        use_gpu = config.get("system", {}).get("use_gpu", False)
        tf_version = config.get("system", {}).get("tensorflow_version", "2.21.0")
        
        print(f"Configuring TensorFlow {tf_version} (source: {tf_source}, GPU: {use_gpu})")
        
        if tf_source == "wheel":
            wheel_path = config.get("system", {}).get("tensorflow_wheel_path")
            if wheel_path and Path(wheel_path).exists():
                print(f"Loading TensorFlow from custom wheel: {wheel_path}")
                # Add wheel to path
                sys.path.insert(0, str(Path(wheel_path).parent))
        elif tf_source == "system":
            print("Using system TensorFlow installation")
        else:
            print(f"Using pip-installed TensorFlow {tf_version}")
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Import TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU available: {bool(tf.config.list_physical_devices('GPU'))}")
        
        # Configure memory growth for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU memory configuration error: {e}")
        
        return tf
    except ImportError as e:
        print(f"Error importing TensorFlow: {e}")
        print("\nPlease check your TensorFlow configuration in config.json")
        print("Options:")
        print("  - Set 'tensorflow_source' to 'pip' for pip installation")
        print("  - Set 'tensorflow_source' to 'wheel' and provide a wheel path")
        print("  - Set 'tensorflow_source' to 'system' if TensorFlow is installed system-wide")
        print("\nTo install TensorFlow 2.21:")
        print("  CPU: pip install tensorflow-cpu==2.21.0")
        print("  GPU: pip install tensorflow==2.21.0")
        sys.exit(1)

# Setup TensorFlow before importing other modules
tf = setup_tensorflow()

# Configure Keras backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import Keras
try:
    import keras
    print(f"Keras version: {keras.__version__}")
except ImportError as e:
    print(f"Error importing Keras: {e}")
    print("Installing Keras 3...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keras>=3.0.0"])
    import keras
    print(f"Keras version: {keras.__version__}")

# Now import other modules
from ui.app import TransformerApp

def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("Transformer TUI Application")
    print("="*60)
    print("Starting Textual interface...")
    print("Use arrow keys to navigate, Enter to select, Ctrl+C to quit")
    print("="*60 + "\n")
    
    app = TransformerApp()
    app.run()

if __name__ == "__main__":
    main()
