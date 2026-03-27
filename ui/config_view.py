from textual.widgets import DataTable, Button, Input, Label, Select, TextArea, TabbedContent, TabPane
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.screen import Screen
from textual import on
import json
from pathlib import Path
from typing import Dict, Any, List, Union


class ConfigEditor(Vertical):
    """Interactive configuration editor with tabs for different sections."""
    
    DEFAULT_CSS = """
    ConfigEditor {
        height: 100%;
    }
    
    #system_fields, #model_fields, #data_fields, #training_fields, #paths_fields {
        height: 100%;
        padding: 1;
    }
    
    .config-row {
        margin: 1 0;
    }
    
    Label {
        margin: 0 1;
    }
    
    Input, Select {
        margin: 0 1;
        width: 100%;
    }
    
    #button_row {
        height: 3;
        margin: 1 0;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.config = None
        self.current_config = None
    
    def compose(self):
        yield Label("Configuration Editor", classes="title")
        
        with Horizontal(id="button_row"):
            yield Button("Load Config", id="load_config", variant="primary")
            yield Button("Save Config", id="save_config", variant="success")
            yield Button("Reset to Default", id="reset_config", variant="warning")
            yield Button("Apply and Reinit Model", id="apply_config", variant="primary")
        
        with TabbedContent():
            with TabPane("System Settings", id="system_tab"):
                with ScrollableContainer(id="system_fields"):
                    yield from self._create_system_config_fields()
            
            with TabPane("Model Architecture", id="model_tab"):
                with ScrollableContainer(id="model_fields"):
                    yield from self._create_model_config_fields()
            
            with TabPane("Data Settings", id="data_tab"):
                with ScrollableContainer(id="data_fields"):
                    yield from self._create_data_config_fields()
            
            with TabPane("Training Settings", id="training_tab"):
                with ScrollableContainer(id="training_fields"):
                    yield from self._create_training_config_fields()
            
            with TabPane("Paths", id="paths_tab"):
                with ScrollableContainer(id="paths_fields"):
                    yield from self._create_paths_config_fields()
            
            with TabPane("Raw JSON", id="json_tab"):
                yield TextArea(id="raw_json_editor", language="json")
        
        yield Label("", id="config_status", classes="status_label")
    
    def _create_system_config_fields(self):
        """Create system configuration input fields."""
        # TensorFlow source dropdown
        yield Label("TensorFlow Source:")
        yield Select(
            [("pip", "pip"), ("wheel", "wheel"), ("system", "system")],
            id="tf_source",
            prompt="Select TensorFlow source"
        )
        
        yield Label("TensorFlow Wheel Path (if using wheel):")
        yield Input(placeholder="/path/to/tensorflow.whl", id="tf_wheel_path")
        
        yield Label("TensorFlow Version:")
        yield Input(value="2.21.0", id="tf_version")
        
        yield Label("Use GPU:")
        yield Select([("true", "Yes"), ("false", "No")], id="use_gpu")
        
        yield Label("Cache Directory:")
        yield Input(value="./cache", id="cache_dir")
    
    def _create_model_config_fields(self):
        """Create model architecture configuration fields."""
        yield Label("Model Type:")
        yield Select(
            [("transformer", "Transformer (Seq2Seq)"), 
             ("transformer_qa", "Transformer QA"),
             ("bert_like", "BERT-like")],
            id="model_type",
            prompt="Select model type"
        )
        
        yield Label("QA Type (for Transformer QA):")
        yield Select(
            [("extractive", "Extractive QA"),
             ("abstractive", "Abstractive QA"),
             ("multiple_choice", "Multiple Choice QA")],
            id="qa_type",
            prompt="Select QA type",
            allow_blank=True
        )
        
        yield Label("Vocab Size:")
        yield Input(value="10000", id="vocab_size")
        
        yield Label("Max Sequence Length:")
        yield Input(value="100", id="max_len")
        
        yield Label("Model Dimension (d_model):")
        yield Input(value="512", id="d_model")
        
        yield Label("Number of Attention Heads:")
        yield Input(value="8", id="num_heads")
        
        yield Label("Feed-forward Dimension (dff):")
        yield Input(value="2048", id="dff")
        
        yield Label("Number of Layers:")
        yield Input(value="6", id="num_layers")
        
        yield Label("Dropout Rate:")
        yield Input(value="0.1", id="dropout_rate")
        
        yield Label("Learning Rate:")
        yield Input(value="0.001", id="learning_rate")
        
        yield Label("Batch Size:")
        yield Input(value="32", id="batch_size")
        
        yield Label("Epochs:")
        yield Input(value="100", id="epochs")
        
        yield Label("Precision:")
        yield Select([("fp32", "FP32"), ("fp16", "FP16")], id="precision")
    
    def _create_data_config_fields(self):
        """Create data configuration fields."""
        yield Label("Train Split Ratio:")
        yield Input(value="0.8", id="train_split")
        
        yield Label("Validation Split Ratio:")
        yield Input(value="0.1", id="val_split")
        
        yield Label("Test Split Ratio:")
        yield Input(value="0.1", id="test_split")
        
        yield Label("Max Samples (null for all):")
        yield Input(placeholder="null", id="max_samples")
        
        yield Label("Cache Preprocessed Data:")
        yield Select([("true", "Yes"), ("false", "No")], id="cache_preprocessed")
    
    def _create_training_config_fields(self):
        """Create training configuration fields."""
        yield Label("Checkpoint Monitor:")
        yield Select(
            [("val_loss", "Validation Loss"),
             ("val_accuracy", "Validation Accuracy"),
             ("loss", "Training Loss"),
             ("accuracy", "Training Accuracy")],
            id="checkpoint_monitor"
        )
        
        yield Label("Checkpoint Mode:")
        yield Select(
            [("min", "Minimize"), ("max", "Maximize")],
            id="checkpoint_mode"
        )
        
        yield Label("Early Stopping Patience:")
        yield Input(value="10", id="early_stopping_patience")
        
        yield Label("Reduce LR Patience:")
        yield Input(value="5", id="reduce_lr_patience")
        
        yield Label("Reduce LR Factor:")
        yield Input(value="0.5", id="reduce_lr_factor")
        
        yield Label("Save Best Only:")
        yield Select([("true", "Yes"), ("false", "No")], id="save_best_only")
    
    def _create_paths_config_fields(self):
        """Create paths configuration fields."""
        yield Label("Output Directory:")
        yield Input(value="./outputs", id="output_dir")
        
        yield Label("Preprocessed Data Directory:")
        yield Input(value="./outputs/preprocessed", id="preprocessed_dir")
        
        yield Label("Models Directory:")
        yield Input(value="./outputs/models", id="models_dir")
        
        yield Label("Logs Directory:")
        yield Input(value="./outputs/logs", id="logs_dir")
        
        yield Label("History Directory:")
        yield Input(value="./outputs/history", id="history_dir")
        
        yield Label("Attention Maps Directory:")
        yield Input(value="./outputs/attention_maps", id="attention_maps_dir")
    
    def on_mount(self):
        """Load current configuration."""
        self.load_config()
    
    def load_config(self):
        """Load configuration from app."""
        self.config = self.app.config.config
        self.current_config = json.loads(json.dumps(self.config))  # Deep copy
        self.populate_fields()
        self.update_json_editor()
        self.query_one("#config_status").update("Configuration loaded")
    
    def populate_fields(self):
        """Populate all input fields with current config values."""
        # System settings
        self._set_select_value("#tf_source", self.config.get("system", {}).get("tensorflow_source", "pip"))
        self._set_input_value("#tf_wheel_path", self.config.get("system", {}).get("tensorflow_wheel_path", "") or "")
        self._set_input_value("#tf_version", self.config.get("system", {}).get("tensorflow_version", "2.21.0"))
        self._set_select_value("#use_gpu", str(self.config.get("system", {}).get("use_gpu", False)).lower())
        self._set_input_value("#cache_dir", self.config.get("system", {}).get("cache_dir", "./cache"))
        
        # Model settings
        self._set_select_value("#model_type", self.config.get("model", {}).get("model_type", "transformer"))
        self._set_select_value("#qa_type", self.config.get("model", {}).get("qa_type", "") or "")
        self._set_input_value("#vocab_size", str(self.config.get("model", {}).get("vocab_size", 10000)))
        self._set_input_value("#max_len", str(self.config.get("model", {}).get("max_len", 100)))
        self._set_input_value("#d_model", str(self.config.get("model", {}).get("d_model", 512)))
        self._set_input_value("#num_heads", str(self.config.get("model", {}).get("num_heads", 8)))
        self._set_input_value("#dff", str(self.config.get("model", {}).get("dff", 2048)))
        self._set_input_value("#num_layers", str(self.config.get("model", {}).get("num_layers", 6)))
        self._set_input_value("#dropout_rate", str(self.config.get("model", {}).get("dropout_rate", 0.1)))
        self._set_input_value("#learning_rate", str(self.config.get("model", {}).get("learning_rate", 0.001)))
        self._set_input_value("#batch_size", str(self.config.get("model", {}).get("batch_size", 32)))
        self._set_input_value("#epochs", str(self.config.get("model", {}).get("epochs", 100)))
        self._set_select_value("#precision", self.config.get("model", {}).get("precision", "fp32"))
        
        # Data settings
        self._set_input_value("#train_split", str(self.config.get("data", {}).get("train_split", 0.8)))
        self._set_input_value("#val_split", str(self.config.get("data", {}).get("val_split", 0.1)))
        self._set_input_value("#test_split", str(self.config.get("data", {}).get("test_split", 0.1)))
        max_samples = self.config.get("data", {}).get("max_samples")
        self._set_input_value("#max_samples", str(max_samples) if max_samples is not None else "")
        self._set_select_value("#cache_preprocessed", str(self.config.get("data", {}).get("cache_preprocessed", True)).lower())
        
        # Training settings
        self._set_select_value("#checkpoint_monitor", self.config.get("training", {}).get("checkpoint_monitor", "val_loss"))
        self._set_select_value("#checkpoint_mode", self.config.get("training", {}).get("checkpoint_mode", "min"))
        self._set_input_value("#early_stopping_patience", str(self.config.get("training", {}).get("early_stopping_patience", 10)))
        self._set_input_value("#reduce_lr_patience", str(self.config.get("training", {}).get("reduce_lr_patience", 5)))
        self._set_input_value("#reduce_lr_factor", str(self.config.get("training", {}).get("reduce_lr_factor", 0.5)))
        self._set_select_value("#save_best_only", str(self.config.get("training", {}).get("save_best_only", True)).lower())
        
        # Paths settings
        self._set_input_value("#output_dir", self.config.get("paths", {}).get("output_dir", "./outputs"))
        self._set_input_value("#preprocessed_dir", self.config.get("paths", {}).get("preprocessed_dir", "./outputs/preprocessed"))
        self._set_input_value("#models_dir", self.config.get("paths", {}).get("models_dir", "./outputs/models"))
        self._set_input_value("#logs_dir", self.config.get("paths", {}).get("logs_dir", "./outputs/logs"))
        self._set_input_value("#history_dir", self.config.get("paths", {}).get("history_dir", "./outputs/history"))
        self._set_input_value("#attention_maps_dir", self.config.get("paths", {}).get("attention_maps_dir", "./outputs/attention_maps"))
    
    def _set_input_value(self, selector: str, value: str):
        """Set input field value."""
        try:
            widget = self.query_one(selector)
            if widget and hasattr(widget, 'value'):
                widget.value = value
        except:
            pass
    
    def _set_select_value(self, selector: str, value: str):
        """Set select field value."""
        try:
            widget = self.query_one(selector)
            if widget and hasattr(widget, 'value'):
                widget.value = value
        except:
            pass
    
    def update_json_editor(self):
        """Update the raw JSON editor with current config."""
        json_editor = self.query_one("#raw_json_editor")
        if json_editor:
            json_editor.text = json.dumps(self.current_config, indent=2)
    
    def collect_config_from_fields(self) -> Dict[str, Any]:
        """Collect configuration from all input fields."""
        config = {
            "system": {
                "tensorflow_source": self._get_select_value("#tf_source"),
                "tensorflow_wheel_path": self._get_input_value("#tf_wheel_path") or None,
                "tensorflow_version": self._get_input_value("#tf_version"),
                "use_gpu": self._get_select_value("#use_gpu") == "true",
                "cache_dir": self._get_input_value("#cache_dir")
            },
            "model": {
                "model_type": self._get_select_value("#model_type"),
                "qa_type": self._get_select_value("#qa_type") or None,
                "vocab_size": int(self._get_input_value("#vocab_size")),
                "max_len": int(self._get_input_value("#max_len")),
                "d_model": int(self._get_input_value("#d_model")),
                "num_heads": int(self._get_input_value("#num_heads")),
                "dff": int(self._get_input_value("#dff")),
                "num_layers": int(self._get_input_value("#num_layers")),
                "dropout_rate": float(self._get_input_value("#dropout_rate")),
                "learning_rate": float(self._get_input_value("#learning_rate")),
                "batch_size": int(self._get_input_value("#batch_size")),
                "epochs": int(self._get_input_value("#epochs")),
                "precision": self._get_select_value("#precision")
            },
            "data": {
                "train_split": float(self._get_input_value("#train_split")),
                "val_split": float(self._get_input_value("#val_split")),
                "test_split": float(self._get_input_value("#test_split")),
                "max_samples": int(self._get_input_value("#max_samples")) if self._get_input_value("#max_samples") not in ["", "null", "None"] else None,
                "cache_preprocessed": self._get_select_value("#cache_preprocessed") == "true"
            },
            "training": {
                "checkpoint_monitor": self._get_select_value("#checkpoint_monitor"),
                "checkpoint_mode": self._get_select_value("#checkpoint_mode"),
                "early_stopping_patience": int(self._get_input_value("#early_stopping_patience")),
                "reduce_lr_patience": int(self._get_input_value("#reduce_lr_patience")),
                "reduce_lr_factor": float(self._get_input_value("#reduce_lr_factor")),
                "save_best_only": self._get_select_value("#save_best_only") == "true"
            },
            "paths": {
                "output_dir": self._get_input_value("#output_dir"),
                "preprocessed_dir": self._get_input_value("#preprocessed_dir"),
                "models_dir": self._get_input_value("#models_dir"),
                "logs_dir": self._get_input_value("#logs_dir"),
                "history_dir": self._get_input_value("#history_dir"),
                "attention_maps_dir": self._get_input_value("#attention_maps_dir")
            },
            "visualization": {
                "attention_head_size": 10,
                "heatmap_colormap": "viridis",
                "save_attention_maps": True
            }
        }
        
        return config
    
    def _get_input_value(self, selector: str) -> str:
        """Get input field value."""
        try:
            widget = self.query_one(selector)
            return widget.value if widget and hasattr(widget, 'value') else ""
        except:
            return ""
    
    def _get_select_value(self, selector: str) -> str:
        """Get select field value."""
        try:
            widget = self.query_one(selector)
            return widget.value if widget and hasattr(widget, 'value') else ""
        except:
            return ""
    
    @on(Button.Pressed, "#load_config")
    def on_load_config(self):
        """Load configuration from file."""
        self.load_config()
        self.app.notify("Configuration loaded")
    
    @on(Button.Pressed, "#save_config")
    def on_save_config(self):
        """Save configuration to file."""
        try:
            # Collect from fields
            self.current_config = self.collect_config_from_fields()
            
            # Save to app config
            self.app.config.config = self.current_config
            self.app.config.save_config(self.current_config)
            
            # Update JSON editor
            self.update_json_editor()
            
            self.query_one("#config_status").update("Configuration saved successfully")
            self.app.notify("Configuration saved successfully")
        except Exception as e:
            self.query_one("#config_status").update(f"Error saving config: {str(e)}")
            self.app.notify(f"Error saving config: {str(e)}", severity="error")
    
    @on(Button.Pressed, "#reset_config")
    def on_reset_config(self):
        """Reset configuration to defaults."""
        default_config = {
            "system": {
                "tensorflow_source": "pip",
                "tensorflow_wheel_path": None,
                "tensorflow_version": "2.21.0",
                "use_gpu": False,
                "cache_dir": "./cache"
            },
            "model": {
                "model_type": "transformer",
                "qa_type": None,
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
                "max_samples": None,
                "cache_preprocessed": True
            },
            "training": {
                "checkpoint_monitor": "val_loss",
                "checkpoint_mode": "min",
                "early_stopping_patience": 10,
                "reduce_lr_patience": 5,
                "reduce_lr_factor": 0.5,
                "save_best_only": True
            },
            "paths": {
                "output_dir": "./outputs",
                "preprocessed_dir": "./outputs/preprocessed",
                "models_dir": "./outputs/models",
                "logs_dir": "./outputs/logs",
                "history_dir": "./outputs/history",
                "attention_maps_dir": "./outputs/attention_maps"
            },
            "visualization": {
                "attention_head_size": 10,
                "heatmap_colormap": "viridis",
                "save_attention_maps": True
            }
        }
        
        self.current_config = default_config
        self.populate_fields()
        self.update_json_editor()
        self.query_one("#config_status").update("Reset to default configuration")
        self.app.notify("Reset to default configuration")
    
    @on(Button.Pressed, "#apply_config")
    def on_apply_config(self):
        """Apply configuration and reinitialize model."""
        try:
            # Save current config
            self.current_config = self.collect_config_from_fields()
            self.app.config.config = self.current_config
            self.app.config.save_config(self.current_config)
            
            # Reinitialize model with new config
            self.app._create_model()
            
            # Refresh model summary view
            try:
                model_summary = self.app.query_one("#summary")
                if hasattr(model_summary, 'load_model_summary'):
                    model_summary.load_model_summary()
            except:
                pass
            
            self.query_one("#config_status").update("Configuration applied and model reinitialized")
            self.app.notify("Configuration applied and model reinitialized successfully")
            
        except Exception as e:
            self.query_one("#config_status").update(f"Error applying config: {str(e)}")
            self.app.notify(f"Error applying config: {str(e)}", severity="error")
    
    @on(TextArea.Changed, "#raw_json_editor")
    def on_json_edit(self, event):
        """Handle raw JSON editing."""
        try:
            edited_json = json.loads(event.text_area.text)
            self.current_config = edited_json
            self.populate_fields()
            self.query_one("#config_status").update("JSON edited - click Save to apply")
        except json.JSONDecodeError:
            self.query_one("#config_status").update("Invalid JSON format")


class ModelInitializationScreen(Screen):
    """Screen for initializing a new model."""
    
    DEFAULT_CSS = """
    ModelInitializationScreen {
        align: center middle;
    }
    
    #model_init_form {
        width: 80;
        height: auto;
        border: solid $primary;
        padding: 2;
        background: $surface;
    }
    
    .config-row {
        margin: 1 0;
    }
    
    Button {
        margin: 1 1;
    }
    """
    
    def compose(self):
        yield Container(
            Label("Initialize New Model", classes="title"),
            
            Container(
                Label("Model Configuration", classes="subtitle"),
                
                Horizontal(
                    Label("Model Type:", classes="config-label"),
                    Select(
                        [("transformer", "Transformer (Seq2Seq)"),
                         ("transformer_qa", "Transformer QA"),
                         ("bert_like", "BERT-like")],
                        id="init_model_type",
                        classes="config-select"
                    ),
                    classes="config-row"
                ),
                
                Horizontal(
                    Label("QA Type (for Transformer QA):", classes="config-label"),
                    Select(
                        [("extractive", "Extractive QA"),
                         ("abstractive", "Abstractive QA"),
                         ("multiple_choice", "Multiple Choice QA")],
                        id="init_qa_type",
                        allow_blank=True,
                        classes="config-select"
                    ),
                    classes="config-row"
                ),
                
                Horizontal(
                    Label("Vocab Size:", classes="config-label"),
                    Input(value="10000", id="init_vocab_size", classes="config-input"),
                    classes="config-row"
                ),
                
                Horizontal(
                    Label("Max Sequence Length:", classes="config-label"),
                    Input(value="100", id="init_max_len", classes="config-input"),
                    classes="config-row"
                ),
                
                Horizontal(
                    Label("Model Dimension:", classes="config-label"),
                    Input(value="512", id="init_d_model", classes="config-input"),
                    classes="config-row"
                ),
                
                Horizontal(
                    Label("Number of Heads:", classes="config-label"),
                    Input(value="8", id="init_num_heads", classes="config-input"),
                    classes="config-row"
                ),
                
                Horizontal(
                    Label("Feed-forward Dimension:", classes="config-label"),
                    Input(value="2048", id="init_dff", classes="config-input"),
                    classes="config-row"
                ),
                
                Horizontal(
                    Label("Number of Layers:", classes="config-label"),
                    Input(value="6", id="init_num_layers", classes="config-input"),
                    classes="config-row"
                ),
                
                Horizontal(
                    Label("Dropout Rate:", classes="config-label"),
                    Input(value="0.1", id="init_dropout", classes="config-input"),
                    classes="config-row"
                ),
                
                Horizontal(
                    Button("Initialize Model", id="init_model", variant="primary"),
                    Button("Cancel", id="cancel_init", variant="default"),
                    classes="config-row"
                ),
                
                Label("", id="init_status"),
                id="model_init_form"
            )
        )
    
    @on(Button.Pressed, "#init_model")
    def on_init_model(self):
        """Initialize the model with selected parameters."""
        try:
            # Get values
            model_type = self.query_one("#init_model_type").value
            qa_type = self.query_one("#init_qa_type").value
            vocab_size = int(self.query_one("#init_vocab_size").value)
            max_len = int(self.query_one("#init_max_len").value)
            d_model = int(self.query_one("#init_d_model").value)
            num_heads = int(self.query_one("#init_num_heads").value)
            dff = int(self.query_one("#init_dff").value)
            num_layers = int(self.query_one("#init_num_layers").value)
            dropout = float(self.query_one("#init_dropout").value)
            
            # Update config
            self.app.config.set('model.model_type', model_type)
            self.app.config.set('model.qa_type', qa_type if qa_type else None)
            self.app.config.set('model.vocab_size', vocab_size)
            self.app.config.set('model.max_len', max_len)
            self.app.config.set('model.d_model', d_model)
            self.app.config.set('model.num_heads', num_heads)
            self.app.config.set('model.dff', dff)
            self.app.config.set('model.num_layers', num_layers)
            self.app.config.set('model.dropout_rate', dropout)
            
            # Reinitialize model
            self.app._create_model()
            
            # Update status
            self.query_one("#init_status").update("Model initialized successfully!")
            self.app.notify(f"Initialized {model_type} model with {d_model} dimensions")
            
            # Close after 2 seconds
            self.set_timer(2, self.dismiss)
            
        except Exception as e:
            self.query_one("#init_status").update(f"Error: {str(e)}")
            self.app.notify(f"Error initializing model: {str(e)}", severity="error")
    
    @on(Button.Pressed, "#cancel_init")
    def on_cancel(self):
        """Cancel model initialization."""
        self.dismiss()
