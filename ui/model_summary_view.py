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
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.param_counts = {}
    
    def compose(self):
        yield Label("Model Architecture", classes="title")
        
        with Horizontal():
            yield Button("Refresh Summary", id="refresh_summary", variant="primary")
            yield Button("Export Summary", id="export_summary", variant="default")
            yield Button("Save Model Config", id="save_config", variant="default")
        
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
        
        # Display layers table
        self.display_layers_table()
        
        # Display parameters statistics
        self.display_param_stats()
        
        # Display layer details
        self.display_layer_details()
        
        # Display model graph info
        self.display_model_graph()
    
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
                output_shape = str(layer.output_shape)
            except:
                output_shape = "Unknown"
            
            # Get parameter counts
            param_count = layer.count_params()
            trainable = sum([1 for w in layer.trainable_weights])
            is_trainable = "Yes" if trainable > 0 else "No"
            
            if trainable > 0:
                trainable_params += param_count
            else:
                non_trainable_params += param_count
            
            total_params += param_count
            
            # Add row with color coding
            row_color = "green" if trainable > 0 else "yellow"
            layers_table.add_row(
                str(i),
                layer_name,
                layer_type,
                output_shape,
                f"{param_count:,}",
                is_trainable,
                classes=row_color
            )
        
        # Add summary row
        layers_table.add_row(
            "TOTAL",
            "-",
            "-",
            "-",
            f"{total_params:,}",
            f"Trainable: {trainable_params:,}",
            classes="bold"
        )
        
        # Store parameter counts for later use
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
        param_memory_mb = (self.param_counts['total'] * 4) / (1024 * 1024)  # 4 bytes per param (float32)
        params_table.add_row("Estimated Memory (FP32)", f"{param_memory_mb:.2f} MB")
        
        # Add layer type distribution
        layer_types = {}
        for layer in self.model.layers:
            layer_type = layer.__class__.__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        params_table.add_row("", "")
        params_table.add_row("Layer Type Distribution", "")
        for layer_type, count in layer_types.items():
            params_table.add_row(f"  {layer_type}", str(count))
        
        # Add attention layer details if present
        attention_layers = [l for l in self.model.layers if 'Attention' in l.__class__.__name__]
        if attention_layers:
            params_table.add_row("", "")
            params_table.add_row("Attention Layers", f"{len(attention_layers)}")
            for i, layer in enumerate(attention_layers):
                if hasattr(layer, 'num_heads'):
                    params_table.add_row(f"  Layer {i+1}", f"{layer.name} (heads: {layer.num_heads})")
    
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
                details.append(f"  Output shape: {layer.output_shape}")
            except:
                details.append("  Output shape: Unknown")
            
            # Parameters
            param_count = layer.count_params()
            details.append(f"  Parameters: {param_count:,}")
            
            # Trainable weights
            trainable_weights = layer.trainable_weights
            if trainable_weights:
                details.append(f"  Trainable weights: {len(trainable_weights)}")
                for weight in trainable_weights[:3]:  # Show first 3
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
            
            # Layer input/output details
            try:
                if hasattr(layer, 'input'):
                    details.append(f"  Input: {layer.input.shape if hasattr(layer.input, 'shape') else 'Unknown'}")
                if hasattr(layer, 'output'):
                    details.append(f"  Output: {layer.output.shape if hasattr(layer.output, 'shape') else 'Unknown'}")
            except:
                pass
        
        # Add total memory summary
        details.append("\n" + "=" * 80)
        details.append("MEMORY SUMMARY")
        details.append("=" * 80)
        
        # Calculate total memory for each layer
        total_memory = 0
        for layer in self.model.layers:
            param_count = layer.count_params()
            if param_count > 0:
                memory_mb = (param_count * 4) / (1024 * 1024)
                details.append(f"  {layer.name}: {memory_mb:.2f} MB")
                total_memory += memory_mb
        
        details.append(f"\n  Total model memory: {total_memory:.2f} MB")
        details.append(f"  With optimizer states: {total_memory * 2:.2f} MB (approx)")
        
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
                self._print_connections(layer.name, connections, graph, "     ")
        
        # Display all layers with their connections
        graph.append("\n\nComplete Layer Flow:")
        graph.append("-" * 40)
        
        for i, layer in enumerate(self.model.layers):
            graph.append(f"\n[{i}] {layer.name}")
            graph.append(f"    Type: {layer.__class__.__name__}")
            
            # Input connections
            inbound = connections.get(layer.name, [])
            if inbound:
                graph.append(f"    Inputs from: {', '.join(inbound)}")
            
            # Output shape
            try:
                graph.append(f"    Output shape: {layer.output_shape}")
            except:
                pass
            
            # Parameter count
            param_count = layer.count_params()
            graph.append(f"    Parameters: {param_count:,}")
        
        # Add complexity analysis
        graph.append("\n\n" + "=" * 80)
        graph.append("COMPLEXITY ANALYSIS")
        graph.append("=" * 80)
        
        # Calculate FLOPs estimation (rough)
        total_flops = 0
        for layer in self.model.layers:
            param_count = layer.count_params()
            if param_count > 0:
                # Rough estimation: FLOPs ≈ 2 * parameters for forward pass
                flops = param_count * 2
                total_flops += flops
                graph.append(f"  {layer.name}: ~{flops:,} FLOPs")
        
        graph.append(f"\n  Total FLOPs (forward pass): ~{total_flops:,}")
        graph.append(f"  Total FLOPs (forward + backward): ~{total_flops * 3:,}")
        
        graph_text.update("\n".join(graph))
    
    def _print_connections(self, layer_name: str, connections: Dict, graph: List, indent: str):
        """Recursively print layer connections."""
        for next_layer, inputs in connections.items():
            if layer_name in inputs:
                graph.append(f"{indent}├─ {next_layer}")
                self._print_connections(next_layer, connections, graph, indent + "│  ")
    
    def display_no_model_message(self):
        """Display message when no model is available."""
        layers_table = self.query_one("#layers_table")
        layers_table.clear()
        layers_table.add_row("No model loaded", "Please initialize model first")
        
        params_table = self.query_one("#params_table")
        params_table.clear()
        params_table.add_row("Status", "Model not loaded")
        
        details_text = self.query_one("#layer_details_text")
        details_text.update("No model loaded. Please ensure a model is initialized.")
        
        graph_text = self.query_one("#model_graph_text")
        graph_text.update("No model loaded. Please ensure a model is initialized.")
    
    @on(Button.Pressed, "#refresh_summary")
    def refresh_summary(self):
        """Refresh the model summary."""
        self.load_model_summary()
        self.app.notify("Model summary refreshed")
    
    @on(Button.Pressed, "#export_summary")
    def export_summary(self):
        """Export model summary to file."""
        if not self.model:
            self.app.notify("No model to export", severity="error")
            return
        
        # Create export directory if it doesn't exist
        export_dir = Path(self.app.config.get('paths.output_dir')) / "model_summaries"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export as text
        timestamp = __import__('time').strftime('%Y%m%d_%H%M%S')
        txt_path = export_dir / f"model_summary_{timestamp}.txt"
        
        with open(txt_path, 'w') as f:
            # Write model summary
            f.write("=" * 80 + "\n")
            f.write("MODEL SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Write layer details
            for i, layer in enumerate(self.model.layers):
                f.write(f"[{i}] {layer.name} ({layer.__class__.__name__})\n")
                f.write("-" * 60 + "\n")
                
                try:
                    f.write(f"  Output shape: {layer.output_shape}\n")
                except:
                    f.write("  Output shape: Unknown\n")
                
                f.write(f"  Parameters: {layer.count_params():,}\n")
                
                if hasattr(layer, 'units'):
                    f.write(f"  Units: {layer.units}\n")
                if hasattr(layer, 'num_heads'):
                    f.write(f"  Number of heads: {layer.num_heads}\n")
                
                f.write("\n")
            
            # Write parameter summary
            f.write("=" * 80 + "\n")
            f.write("PARAMETER SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Parameters: {self.param_counts['total']:,}\n")
            f.write(f"Trainable Parameters: {self.param_counts['trainable']:,}\n")
            f.write(f"Non-trainable Parameters: {self.param_counts['non_trainable']:,}\n")
        
        self.app.notify(f"Model summary exported to {txt_path}")
    
    @on(Button.Pressed, "#save_config")
    def save_model_config(self):
        """Save model configuration to JSON file."""
        if not self.model:
            self.app.notify("No model to save", severity="error")
            return
        
        # Export config
        export_dir = Path(self.app.config.get('paths.output_dir')) / "model_configs"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = __import__('time').strftime('%Y%m%d_%H%M%S')
        config_path = export_dir / f"model_config_{timestamp}.json"
        
        # Get model config
        try:
            model_config = self.model.get_config()
            
            # Add additional metadata
            model_metadata = {
                'model_type': self.model.__class__.__name__,
                'total_parameters': self.param_counts['total'],
                'trainable_parameters': self.param_counts['trainable'],
                'layer_count': len(self.model.layers),
                'timestamp': timestamp,
                'config': model_config
            }
            
            with open(config_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            self.app.notify(f"Model config saved to {config_path}")
        except Exception as e:
            self.app.notify(f"Error saving config: {e}", severity="error")
    
    def watch_model(self, model):
        """Watch for model changes."""
        self.model = model
        self.load_model_summary()
