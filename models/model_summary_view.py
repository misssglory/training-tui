from textual.widgets import DataTable, Static, Label, Button, TabbedContent, TabPane
from textual.containers import Horizontal, Vertical, Container
from textual import on
from textual.reactive import reactive
import keras
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

class ModelSummaryView(Vertical):
    """Model summary view showing architecture, parameters, and layer details."""
    
    DEFAULT_CSS = """
    ModelSummaryView {
        height: 100%;
    }
    
    #layers_table {
        height: 60%;
    }
    
    #params_table {
        height: 40%;
    }
    
    .title {
        text-style: bold;
        padding: 1;
        background: $primary;
        color: $text;
    }
    
    .subtitle {
        text-style: bold;
        padding: 0 1;
        margin-top: 1;
    }
    
    Button {
        margin: 0 1;
    }
    
    #button_row {
        height: 3;
        margin: 1 0;
    }
    
    TabbedContent {
        height: 100%;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.param_counts = {}
    
    def compose(self):
        yield Label("Model Architecture", classes="title")
        
        with Horizontal(id="button_row"):
            yield Button("Refresh Summary", id="refresh_summary", variant="primary")
            yield Button("Export Summary", id="export_summary", variant="default")
            yield Button("Save Model Config", id="save_config", variant="default")
            yield Button("Build Model", id="build_model", variant="warning")
        
        with TabbedContent():
            with TabPane("Layer Summary", id="layer_summary"):
                yield DataTable(id="layers_table")
            
            with TabPane("Parameter Statistics", id="param_stats"):
                yield DataTable(id="params_table")
            
            with TabPane("Layer Details", id="layer_details"):
                yield Static(id="layer_details_text")
            
            with TabPane("Model Graph", id="model_graph"):
                yield Static(id="model_graph_text")
    
    def on_mount(self):
        """Initialize the view and load model summary."""
        self.setup_tables()
        self.load_model_summary()
    
    def setup_tables(self):
        """Setup the data tables with appropriate columns."""
        # Layers table
        layers_table = self.query_one("#layers_table")
        layers_table.add_columns("Layer #", "Layer Name", "Layer Type", "Output Shape", "Parameters", "Trainable")
        
        # Parameters table
        params_table = self.query_one("#params_table")
        params_table.add_columns("Metric", "Value")
    
    def load_model_summary(self):
        """Load and display model summary."""
        trainer = self.app.get_trainer()
        if trainer and trainer.model:
            self.model = trainer.model
            self.display_model_summary()
        else:
            self.display_no_model_message()
    
    def display_model_summary(self):
        """Display the model summary in the tables."""
        if not self.model:
            return
        
        # Build model if not already built
        if not self.model.built:
            self.build_model_with_dummy_input()
        
        # Display layers table
        self.display_layers_table()
        
        # Display parameters statistics
        self.display_param_stats()
        
        # Display layer details
        self.display_layer_details()
        
        # Display model graph info
        self.display_model_graph()
    
    def build_model_with_dummy_input(self):
        """Build the model with dummy input to initialize layers."""
        try:
            batch_size = self.app.config.get('model.batch_size', 32)
            max_len = self.app.config.get('model.max_len', 100)
            
            # Create dummy inputs
            dummy_input = keras.ops.zeros((batch_size, max_len), dtype='int32')
            dummy_target = keras.ops.zeros((batch_size, max_len), dtype='int32')
            
            # Call the model to build it
            if isinstance(self.model, (Transformer, TransformerQA, BertLikeTransformer)):
                self.model([dummy_input, dummy_target], training=False)
            elif hasattr(self.model, 'call') and hasattr(self.model, 'input_spec'):
                # Try to build with single input if it's a different model type
                try:
                    self.model(dummy_input, training=False)
                except:
                    pass
            
            print(f"Model built successfully with input shape: ({batch_size}, {max_len})")
            
        except Exception as e:
            print(f"Error building model: {e}")
            # Try alternative building method
            try:
                # Build each layer individually
                for layer in self.model.layers:
                    if not layer.built:
                        if hasattr(layer, 'build'):
                            # Determine input shape for the layer
                            if isinstance(layer, keras.layers.Embedding):
                                input_dim = self.app.config.get('model.vocab_size', 10000)
                                output_dim = self.app.config.get('model.d_model', 512)
                                layer.build((None, None))
                            elif hasattr(layer, 'units'):
                                layer.build((None, layer.units))
                            else:
                                layer.build((None, self.app.config.get('model.max_len', 100)))
                print("Model built using layer-by-layer approach")
            except Exception as e2:
                print(f"Error in layer-by-layer build: {e2}")
    
    def display_layers_table(self):
        """Display all layers with their details."""
        layers_table = self.query_one("#layers_table")
        layers_table.clear()
        
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        for i, layer in enumerate(self.model.layers):
            # Get layer info
            layer_name = layer.name
            layer_type = layer.__class__.__name__
            
            # Get output shape
            try:
                if layer.built:
                    output_shape = str(layer.output_shape)
                else:
                    output_shape = "Not built"
            except:
                output_shape = "Unknown"
            
            # Get parameter counts
            try:
                param_count = layer.count_params()
            except ValueError:
                # Layer not built, try to count weights manually
                try:
                    param_count = sum([w.numpy().size for w in layer.weights]) if layer.weights else 0
                except:
                    param_count = 0
            
            trainable = sum([1 for w in layer.trainable_weights])
            is_trainable = "Yes" if trainable > 0 else "No"
            
            if trainable > 0:
                trainable_params += param_count
            else:
                non_trainable_params += param_count
            
            total_params += param_count
            
            # Add row
            layers_table.add_row(
                str(i),
                layer_name,
                layer_type,
                output_shape,
                f"{param_count:,}" if param_count > 0 else "0",
                is_trainable
            )
        
        # Add summary row
        layers_table.add_row(
            "TOTAL",
            "-",
            "-",
            "-",
            f"{total_params:,}",
            f"Trainable: {trainable_params:,}"
        )
        
        # Store parameter counts
        self.param_counts = {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    def display_param_stats(self):
        """Display parameter statistics."""
        params_table = self.query_one("#params_table")
        params_table.clear()
        
        if not self.param_counts:
            return
        
        # Add parameter statistics
        params_table.add_row("Total Parameters", f"{self.param_counts['total']:,}")
        params_table.add_row("Trainable Parameters", f"{self.param_counts['trainable']:,}")
        params_table.add_row("Non-trainable Parameters", f"{self.param_counts['non_trainable']:,}")
        
        # Memory estimation
        param_memory_mb = (self.param_counts['total'] * 4) / (1024 * 1024)
        params_table.add_row("Estimated Memory (FP32)", f"{param_memory_mb:.2f} MB")
        
        # Add layer type distribution
        layer_types = {}
        for layer in self.model.layers:
            layer_type = layer.__class__.__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        if layer_types:
            params_table.add_row("", "")
            params_table.add_row("Layer Type Distribution", "")
            for layer_type, count in layer_types.items():
                params_table.add_row(f"  {layer_type}", str(count))
        
        # Add model type info
        params_table.add_row("", "")
        params_table.add_row("Model Information", "")
        params_table.add_row("  Model Class", self.model.__class__.__name__)
        params_table.add_row("  Built", str(self.model.built))
    
    def display_layer_details(self):
        """Display detailed information about each layer."""
        details_text = self.query_one("#layer_details_text")
        
        if not self.model:
            details_text.update("No model loaded")
            return
        
        details = []
        details.append("=" * 80)
        details.append("LAYER DETAILS")
        details.append("=" * 80)
        
        for i, layer in enumerate(self.model.layers):
            details.append(f"\n[{i}] {layer.name} ({layer.__class__.__name__})")
            details.append("-" * 60)
            
            # Output shape
            try:
                if layer.built:
                    details.append(f"  Output shape: {layer.output_shape}")
                else:
                    details.append("  Output shape: Not built")
            except:
                details.append("  Output shape: Unknown")
            
            # Parameters
            try:
                param_count = layer.count_params()
                details.append(f"  Parameters: {param_count:,}")
            except:
                details.append("  Parameters: Not available (layer not built)")
            
            # Built status
            details.append(f"  Built: {layer.built}")
            
            # Trainable weights
            trainable_weights = layer.trainable_weights
            if trainable_weights:
                details.append(f"  Trainable weights: {len(trainable_weights)}")
                for weight in trainable_weights[:3]:
                    details.append(f"    - {weight.name}: {weight.shape}")
            
            # Non-trainable weights
            non_trainable_weights = layer.non_trainable_weights
            if non_trainable_weights:
                details.append(f"  Non-trainable weights: {len(non_trainable_weights)}")
            
            # Layer-specific configuration
            if hasattr(layer, 'units'):
                details.append(f"  Units: {layer.units}")
            if hasattr(layer, 'num_heads'):
                details.append(f"  Number of heads: {layer.num_heads}")
            if hasattr(layer, 'd_model'):
                details.append(f"  Model dimension: {layer.d_model}")
            if hasattr(layer, 'dropout'):
                details.append(f"  Dropout rate: {layer.dropout}")
            if hasattr(layer, 'activation'):
                details.append(f"  Activation: {layer.activation}")
            if hasattr(layer, 'vocab_size'):
                details.append(f"  Vocab size: {layer.vocab_size}")
        
        details_text.update("\n".join(details))
    
    def display_model_graph(self):
        """Display a text-based representation of the model graph."""
        graph_text = self.query_one("#model_graph_text")
        
        if not self.model:
            graph_text.update("No model loaded")
            return
        
        graph = []
        graph.append("=" * 80)
        graph.append("MODEL GRAPH (Text Representation)")
        graph.append("=" * 80)
        graph.append("")
        
        # Build dependency graph
        connections = {}
        for layer in self.model.layers:
            layer_name = layer.name
            connections[layer_name] = []
            
            # Try to find connections
            try:
                if hasattr(layer, '_inbound_nodes'):
                    for node in layer._inbound_nodes:
                        if hasattr(node, 'inbound_layers'):
                            for inbound in node.inbound_layers:
                                if hasattr(inbound, 'name'):
                                    connections[layer_name].append(inbound.name)
            except:
                pass
        
        # Display graph in a hierarchical way
        graph.append("Layer Connections:")
        graph.append("-" * 40)
        
        # Find input layers
        input_layers = [l for l in self.model.layers if len(connections.get(l.name, [])) == 0]
        
        if input_layers:
            graph.append("\nInput Layers:")
            for layer in input_layers:
                graph.append(f"  └─ {layer.name} ({layer.__class__.__name__})")
        
        # Display all layers
        graph.append("\n\nComplete Layer List:")
        graph.append("-" * 40)
        
        for i, layer in enumerate(self.model.layers):
            graph.append(f"\n[{i}] {layer.name}")
            graph.append(f"    Type: {layer.__class__.__name__}")
            graph.append(f"    Built: {layer.built}")
            
            # Input connections
            inbound = connections.get(layer.name, [])
            if inbound:
                graph.append(f"    Inputs from: {', '.join(inbound)}")
            
            # Output shape
            try:
                if layer.built:
                    graph.append(f"    Output shape: {layer.output_shape}")
            except:
                pass
            
            # Parameter count
            try:
                param_count = layer.count_params()
                graph.append(f"    Parameters: {param_count:,}")
            except:
                graph.append(f"    Parameters: Not built")
        
        graph_text.update("\n".join(graph))
    
    def display_no_model_message(self):
        """Display message when no model is available."""
        layers_table = self.query_one("#layers_table")
        layers_table.clear()
        layers_table.add_row("No model loaded", "Please initialize model first")
        
        params_table = self.query_one("#params_table")
        params_table.clear()
        params_table.add_row("Status", "Model not loaded")
        params_table.add_row("Action", "Use Configuration tab to initialize model")
        
        details_text = self.query_one("#layer_details_text")
        details_text.update("No model loaded. Please ensure a model is initialized.\n\nSteps:\n1. Go to Configuration tab\n2. Configure model parameters\n3. Click 'Apply and Reinit Model'")
        
        graph_text = self.query_one("#model_graph_text")
        graph_text.update("No model loaded. Model will be built when training starts or when you click 'Build Model'.")
    
    @on(Button.Pressed, "#refresh_summary")
    def refresh_summary(self):
        """Refresh the model summary."""
        self.load_model_summary()
        self.app.notify("Model summary refreshed")
    
    @on(Button.Pressed, "#build_model")
    def build_model(self):
        """Manually build the model."""
        try:
            self.build_model_with_dummy_input()
            self.load_model_summary()
            self.app.notify("Model built successfully!")
        except Exception as e:
            self.app.notify(f"Error building model: {str(e)}", severity="error")
    
    @on(Button.Pressed, "#export_summary")
    def export_summary(self):
        """Export model summary to file."""
        if not self.model:
            self.app.notify("No model to export", severity="error")
            return
        
        # Create export directory
        export_dir = Path(self.app.config.get('paths.output_dir')) / "model_summaries"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export as text
        timestamp = __import__('time').strftime('%Y%m%d_%H%M%S')
        txt_path = export_dir / f"model_summary_{timestamp}.txt"
        
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Write model info
            f.write(f"Model Type: {self.model.__class__.__name__}\n")
            f.write(f"Built: {self.model.built}\n")
            f.write(f"Total Parameters: {self.param_counts.get('total', 0):,}\n\n")
            
            # Write layer details
            for i, layer in enumerate(self.model.layers):
                f.write(f"[{i}] {layer.name} ({layer.__class__.__name__})\n")
                f.write("-" * 60 + "\n")
                
                try:
                    if layer.built:
                        f.write(f"  Output shape: {layer.output_shape}\n")
                except:
                    pass
                
                try:
                    param_count = layer.count_params()
                    f.write(f"  Parameters: {param_count:,}\n")
                except:
                    f.write("  Parameters: Not built\n")
                
                f.write(f"  Built: {layer.built}\n")
                
                if hasattr(layer, 'units'):
                    f.write(f"  Units: {layer.units}\n")
                if hasattr(layer, 'num_heads'):
                    f.write(f"  Number of heads: {layer.num_heads}\n")
                
                f.write("\n")
        
        self.app.notify(f"Model summary exported to {txt_path}")
    
    @on(Button.Pressed, "#save_config")
    def save_model_config(self):
        """Save model configuration to JSON file."""
        if not self.model:
            self.app.notify("No model to save", severity="error")
            return
        
        export_dir = Path(self.app.config.get('paths.output_dir')) / "model_configs"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = __import__('time').strftime('%Y%m%d_%H%M%S')
        config_path = export_dir / f"model_config_{timestamp}.json"
        
        try:
            # Get model config
            model_config = self.model.get_config() if hasattr(self.model, 'get_config') else {}
            
            # Add additional metadata
            model_metadata = {
                'model_type': self.model.__class__.__name__,
                'total_parameters': self.param_counts.get('total', 0),
                'trainable_parameters': self.param_counts.get('trainable', 0),
                'layer_count': len(self.model.layers),
                'built': self.model.built,
                'timestamp': timestamp,
                'config': model_config
            }
            
            with open(config_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            self.app.notify(f"Model config saved to {config_path}")
        except Exception as e:
            self.app.notify(f"Error saving config: {e}", severity="error")


# Import needed for type hints
from models.transformer import Transformer, TransformerQA, BertLikeTransformer
