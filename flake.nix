{
  description = "Transformer TUI development environment with Keras 3, Python 3.13, and TensorFlow 2.21";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    home-manager.url = "github:nix-community/home-manager";
    home-manager.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, home-manager }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Python 3.13
        python = pkgs.python313;
        
        # Create a Python environment with system packages
        pythonWithLibs = pkgs.python313.withPackages (ps: with ps; [
          uv
          pkgs.zeromq
          pkgs.stdenv.cc.cc.lib
          # Development tools
          pip
          setuptools
          wheel
        ]);
        
        # Home Manager configuration
        homeConfig = home-manager.lib.homeManagerConfiguration {
          inherit pkgs;
          modules = [
            {
              home = {
                username = builtins.getEnv "USER";
                homeDirectory = builtins.getEnv "HOME";
                stateVersion = "25.11";
              };
              
              programs.vscode = {
                enable = true;
                package = pkgs.vscodium-fhs;
                
                profiles.default.extensions = with pkgs.vscode-extensions; [
                  ms-python.python
                  ms-python.vscode-pylance
                  ms-toolsai.jupyter
                  ms-toolsai.jupyter-keymap
                  ms-toolsai.jupyter-renderers
                  ms-toolsai.vscode-jupyter-cell-tags
                  ms-toolsai.vscode-jupyter-slideshow
                  ms-toolsai.vscode-tensorboard
                  ms-python.black-formatter
                  ms-python.isort
                  charliermarsh.ruff
                  eamodio.gitlens
                  redhat.vscode-yaml
                ];
                
                userSettings = {
                  "python.defaultInterpreterPath" = "./.venv/bin/python";
                  "python.terminal.activateEnvironment" = true;
                  "python.terminal.activateEnvInCurrentTerminal" = true;
                  "python.terminal.launchArgs" = [
                    "--no-warnings"
                  ];
                  "editor.formatOnSave" = true;
                  "editor.defaultFormatter" = "ms-python.black-formatter";
                  "[python]" = {
                    "editor.formatOnSave" = true;
                    "editor.codeActionsOnSave" = {
                      "source.organizeImports" = true;
                    };
                    "editor.defaultFormatter" = "ms-python.black-formatter";
                  };
                  "black-formatter.args" = ["--line-length" "88"];
                  "black-formatter.importStrategy" = "fromEnvironment";
                  "ruff.enable" = true;
                  "jupyter.alwaysTrustNotebooks" = true;
                };
              };
            }
          ];
        };
        
        # Function to create Python environment with appropriate TensorFlow
        createPythonEnv = { useGPU ? false, wheelPath ? null }:
          pkgs.mkShell {
            buildInputs = with pkgs; [
              pythonWithLibs
              uv
              git
              gcc
              stdenv.cc.cc.lib
              zeromq
              libsodium
              libffi
              openssl
              # Graphics and display
              libGL
              libGLU
              libxcb
              xorg.libX11
              xorg.libXext
              xorg.libXrender
              # For visualizations
              imagemagick
              # System tools
              procps
              htop
            ] ++ (if useGPU then [ 
            ] else []);
            
            shellHook = ''
              # Set up library paths
              export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zeromq}/lib:${pkgs.libsodium}/lib:${pkgs.libffi}/lib:${pkgs.openssl}/lib:${pkgs.libGL}/lib:${pkgs.libGLU}/lib:$LD_LIBRARY_PATH
              
              
              # Create and activate uv virtual environment
              if [ ! -d ".venv" ]; then
                echo "Creating uv virtual environment with Python 3.13..."
                uv venv --python ${python}/bin/python .venv
              fi
              
              # Activate virtual environment
              source .venv/bin/activate
              
              # Upgrade pip and basic tools
              echo "Upgrading pip and basic tools..."
              uv pip install --upgrade pip setuptools wheel
              
              # Install base dependencies

              echo $TF_WHEEL
              echo $TF_SOURCE
              echo "Installing base dependencies..."
              uv pip install textual numpy loguru psutil matplotlib pillow scikit-learn pandas pyperclip
              
              # Install Keras 3
              echo "Installing Keras 3..."
              uv pip install keras>=3.0.0
              
              # Install TensorFlow based on configuration
              echo "Installing TensorFlow 2.21..."
              if [ -f "config.json" ]; then
                TF_SOURCE=$(python -c 'import json; print(json.load(open("config.json")).get("system", {}).get("tensorflow_source", "pip"))' )
                TF_WHEEL=$(python -c 'import json; print(json.load(open("config.json")).get("system", {}).get("tensorflow_wheel_path", ""))')
                USE_GPU_CONFIG=$(python -c 'import json; print(json.load(open("config.json")).get("system", {}).get("use_gpu", False))' 2>/dev/null || echo "False")
                echo $TF_WHEEL
                echo $TF_SOURCE
                
                if [ "$TF_SOURCE" = "wheel" ] && [ -n "$TF_WHEEL" ] && [ -f "$TF_WHEEL" ]; then
                
                  echo "Installing TensorFlow from wheel: $TF_WHEEL"
                  uv pip install "$TF_WHEEL"
                elif [ "$TF_SOURCE" = "pip" ]; then
                  if [ "$USE_GPU_CONFIG" = "True" ] || [ "${if useGPU then "true" else "false"}" = "true" ]; then
                    echo "Installing TensorFlow 2.21 with GPU support..."
                    uv pip install tensorflow==2.21.0
                  else
                    echo "Installing TensorFlow 2.21 CPU version..."
                    uv pip install tensorflow-cpu==2.21.0
                  fi
                elif [ "$TF_SOURCE" = "system" ]; then
                  echo "Using system TensorFlow (assuming it's available in PYTHONPATH)"
                else
                  echo "⚠ Unknown TensorFlow source: $TF_SOURCE, installing default TensorFlow 2.21 CPU"
                  uv pip install tensorflow-cpu==2.21.0
                fi
              else
                echo "⚠ No config.json found, installing default TensorFlow 2.21 CPU"
                uv pip install tensorflow-cpu==2.21.0
              fi
              
              # Install additional development tools
              echo "Installing development tools..."
              uv pip install black isort ruff pytest mypy pylint
              
              # Install optional dependencies

              echo "Installing optional dependencies..."
              uv pip install climage jupyter jupyterlab ipywidgets tqdm
              
              # Test imports
              echo ""
              echo "Testing imports..."
              python -c "import numpy; print('  ✓ NumPy:', numpy.__version__)" 2>/dev/null || echo "  ⚠ NumPy import failed"
              python -c "import keras; print('  ✓ Keras:', keras.__version__)" 2>/dev/null || echo "  ⚠ Keras import failed"
              python -c "import tensorflow as tf; print('  ✓ TensorFlow:', tf.__version__)" 2>/dev/null || echo "  ⚠ TensorFlow import failed"
              python -c "import textual; print('  ✓ Textual:', textual.__version__)" 2>/dev/null || echo "  ⚠ Textual import failed"
              
              # Check GPU availability
              python -c "import tensorflow as tf; print(f'  ✓ GPU available: {bool(tf.config.list_physical_devices(\"GPU\"))}')" 2>/dev/null || echo "  ⚠ GPU check failed"
              
              # Create necessary directories
              mkdir -p outputs/{preprocessed,models,logs,history,attention_maps}
              mkdir -p cache
              
              # Set up project-specific VSCodium settings
              mkdir -p .vscode
              cat > .vscode/settings.json << 'EOF'
{
  "python.defaultInterpreterPath": "''${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.terminal.activateEnvInCurrentTerminal": true,
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.pylintEnabled": false,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": "explicit"
  },
  "python.sortImports.args": ["--profile", "black"],
  "python.analysis.extraPaths": ["./"]
}
EOF
              
              # Create .env file with library paths
              cat > .env << EOF
LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zeromq}/lib:${pkgs.libsodium}/lib:${pkgs.libxcb}/lib:$LD_LIBRARY_PATH
PYTHONPATH=.
EOF
              
              echo ""
              echo "╔═══════════════════════════════════════════════════════════════════════════╗"
              echo "║  Transformer TUI Development Environment Ready!                           ║"
              echo "║                                                                           ║"
              echo "║  • Python: $(python --version)                                           ║"
              echo "║  • Virtual env: .venv (activated)                                        ║"
              echo "║  • Python path: $(which python)                                          ║"
              echo "║                                                                           ║"
              echo "║  Framework versions:                                                      ║"
              $(python -c "import keras; print('  ✓ Keras:', keras.__version__)" 2>/dev/null || echo "  ⚠ Keras import failed")
              $(python -c "import tensorflow as tf; print('  ✓ TensorFlow:', tf.__version__)" 2>/dev/null || echo "  ⚠ TensorFlow import failed")
              echo "║                                                                           ║"
              echo "║  Commands:                                                                ║"
              echo "║  • Run app: python main.py                                                ║"
              echo "║  • Run tests: pytest                                                      ║"
              echo "║  • Format code: black . && isort .                                        ║"
              echo "║  • Start VSCodium: codium                                                 ║"
              echo "║                                                                           ║"
              echo "║  TensorFlow 2.21 with Python 3.13                                          ║"
              echo "║  GPU Support: ${if useGPU then "Enabled" else "Disabled"}                                                              ║"
              echo "╚═══════════════════════════════════════════════════════════════════════════╝"
              echo ""
            '';
          };
      in
      {
        # Default dev shell (CPU version)
        devShell = createPythonEnv { useGPU = false; };
        
        # GPU dev shell
        devShellGPU = createPythonEnv { useGPU = true; };
      });
}
