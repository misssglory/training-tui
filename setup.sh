#!/bin/bash

# Setup script for the Transformer TUI application with Python 3.13 and TensorFlow 2.21

set -e

echo "Setting up Transformer TUI application with Python 3.13 and TensorFlow 2.21..."

# Check if Nix is installed
if ! command -v nix &> /dev/null; then
    echo "Nix is not installed. Please install Nix first:"
    echo "curl -L https://nixos.org/nix/install | sh"
    exit 1
fi

# Check if config.json exists
if [ ! -f "config.json" ]; then
    echo "Creating default config.json..."
    cat > config.json << 'EOF'
{
    "system": {
        "tensorflow_source": "pip",
        "tensorflow_wheel_path": null,
        "tensorflow_version": "2.21.0",
        "use_gpu": false,
        "cache_dir": "./cache"
    },
    "model": {
        "vocab_size": 10000,
        "max_len": 100,
        "d_model": 512,
        "num_heads": 8,
        "dff": 2048,
        "num_layers": 6,
        "dropout_rate": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "precision": "fp32"
    },
    "data": {
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "dataset_size": null,
        "max_samples": null,
        "cache_preprocessed": true
    },
    "paths": {
        "output_dir": "./outputs",
        "preprocessed_dir": "./outputs/preprocessed",
        "models_dir": "./outputs/models",
        "logs_dir": "./outputs/logs",
        "history_dir": "./outputs/history",
        "attention_maps_dir": "./outputs/attention_maps"
    },
    "training": {
        "checkpoint_monitor": "val_loss",
        "checkpoint_mode": "min",
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
        "reduce_lr_factor": 0.5,
        "save_best_only": true
    },
    "visualization": {
        "attention_head_size": 10,
        "heatmap_colormap": "viridis",
        "save_attention_maps": true
    }
}
EOF
fi

# Build and enter Nix development environment
echo "Building Nix development environment with Python 3.13 and TensorFlow 2.21..."
if [ "$1" == "gpu" ]; then
    echo "Using GPU-enabled environment..."
    nix develop .#devShellGPU
else
    echo "Using CPU environment..."
    nix develop .#devShell
fi
