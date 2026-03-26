# Transformer TUI Application

A comprehensive Textual-based terminal user interface for training and testing transformer models with Keras 3.

## Features

- 📊 **Interactive TUI**: Full terminal-based interface using Textual
- 🤖 **Transformer Models**: Implementation of "Attention is All You Need" architecture
- 🎯 **Multi-threaded Training**: Train models while interacting with the UI
- 📈 **Real-time Monitoring**: View training logs, metrics, and attention maps
- 💬 **Chat Interface**: Test model predictions during training
- 🔍 **Attention Visualization**: View attention head heatmaps
- 💾 **Memory Management**: Dataset size control and memory monitoring
- ⚙️ **Flexible Configuration**: JSON-based configuration with UI editor
- 🐍 **Reproducible Environment**: Nix flake for consistent development

## Requirements

- Nix package manager (for reproducible environment)
- Python 3.11+
- CUDA-capable GPU (optional, for GPU acceleration)

## Quick Start

### Using Nix (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd transformer-tui
