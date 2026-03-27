from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, TabbedContent, TabPane, Label
from textual import on
from pathlib import Path
import sys
import os

# Import UI components
from ui.config_view import ConfigEditor, ModelInitializationScreen
from ui.dataset_view import DatasetView
from ui.training_view import TrainingView
from ui.model_summary_view import ModelSummaryView
from ui.chat_view import ChatView

# Import core modules
from config import Config
from models.transformer import Transformer, TransformerQA, BertLikeTransformer, create_qa_model
from data.preprocessing import TextPreprocessor
from data.dataset import TextDataset
from training.trainer import Trainer
from utils.logger import setup_logger


class TransformerApp(App):
    """Main application class for the Transformer TUI."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    TabbedContent {
        height: 100%;
    }
    
    TabPane {
        padding: 1;
    }
    
    .title {
        text-style: bold;
        padding: 1;
        background: $primary;
        color: $text;
    }
    
    .status_label {
        padding: 0 1;
    }
    
    .status_value {
        text-style: bold;
    }
    
    #button_row {
        height: 3;
        margin: 1 0;
    }
    
    Button {
        margin: 0 1;
    }
    
    Input {
        margin: 0 1;
    }
    
    Select {
        margin: 0 1;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.logger = setup_logger(self.config)
        
        # Initialize components
        self.preprocessor = TextPreprocessor(
            vocab_size=self.config.get('model.vocab_size', 10000),
            max_len=self.config.get('model.max_len', 100)
        )
        self.dataset = TextDataset(self.config, self.preprocessor)
        self.model = None
        self.trainer = None
        
        # Create initial model
        self._create_model()
    
    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header()
        
        with TabbedContent():
            with TabPane("Configuration", id="config"):
                yield ConfigEditor()
            
            with TabPane("Dataset", id="dataset"):
                yield DatasetView()
            
            with TabPane("Model Summary", id="summary"):
                yield ModelSummaryView()
            
            with TabPane("Training", id="training"):
                yield TrainingView()
            
            with TabPane("Chat", id="chat"):
                yield ChatView()
        
        yield Footer()
    
    def on_mount(self):
        """Called when the app is mounted."""
        self.logger.info("Transformer TUI Application Started")
        self.logger.info(f"TensorFlow version: {self._get_tf_version()}")
        self.logger.info(f"Keras version: {self._get_keras_version()}")
        
        if self.model:
            self.logger.info(f"Model: {self.model.__class__.__name__}")
            self.logger.info(f"Model built: {self.model.built}")
    
    def _get_tf_version(self):
        """Get TensorFlow version."""
        try:
            import tensorflow as tf
            return tf.__version__
        except:
            return "Not installed"
    
    def _get_keras_version(self):
        """Get Keras version."""
        try:
            import keras
            return keras.__version__
        except:
            return "Not installed"
    
    def _create_model(self):
        """Create model based on current configuration."""
        model_config = self.config.get('model')
        model_type = model_config.get('model_type', 'transformer')
        
        # Get common parameters
        vocab_size = model_config.get('vocab_size', 10000)
        max_len = model_config.get('max_len', 100)
        d_model = model_config.get('d_model', 512)
        num_heads = model_config.get('num_heads', 8)
        dff = model_config.get('dff', 2048)
        num_layers = model_config.get('num_layers', 6)
        dropout_rate = model_config.get('dropout_rate', 0.1)
        
        try:
            # Create model based on type
            if model_type == 'transformer':
                self.model = Transformer(
                    vocab_size=vocab_size,
                    max_len=max_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    dff=dff,
                    num_layers=num_layers,
                    dropout_rate=dropout_rate
                )
                # Don't try to count parameters until built
                self.logger.info(f"Created Transformer model (will be built during first use)")
                
            elif model_type == 'transformer_qa':
                qa_type = model_config.get('qa_type', 'extractive')
                self.model = create_qa_model(
                    model_type='transformer_qa',
                    vocab_size=vocab_size,
                    max_len=max_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    dff=dff,
                    num_layers=num_layers,
                    dropout_rate=dropout_rate,
                    qa_type=qa_type
                )
                self.logger.info(f"Created Transformer QA model ({qa_type})")
                
            elif model_type == 'bert_like':
                self.model = BertLikeTransformer(
                    vocab_size=vocab_size,
                    max_len=max_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    dff=dff,
                    num_layers=num_layers,
                    dropout_rate=dropout_rate
                )
                self.logger.info(f"Created BERT-like model")
            
            # Update preprocessor
            self.preprocessor.vocab_size = vocab_size
            self.preprocessor.max_len = max_len
            
            # Create trainer (model may be built during training)
            self.trainer = Trainer(self.config, self.model, self.dataset, self.preprocessor)
            
            # Refresh UI if mounted
            if hasattr(self, 'is_mounted') and self.is_mounted:
                self.refresh_model_views()
                
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def refresh_model_views(self):
        """Refresh any views that display model information."""
        try:
            # Refresh model summary view
            model_summary = self.query_one("#summary")
            if hasattr(model_summary, 'load_model_summary'):
                model_summary.load_model_summary()
        except Exception as e:
            self.logger.debug(f"Could not refresh model summary: {e}")
        
        try:
            # Refresh training view
            training_view = self.query_one("#training")
            if hasattr(training_view, 'update_model_info'):
                training_view.update_model_info()
        except Exception as e:
            self.logger.debug(f"Could not refresh training view: {e}")
        
        try:
            # Refresh chat view
            chat_view = self.query_one("#chat")
            if hasattr(chat_view, 'update_status'):
                chat_view.update_status()
        except Exception as e:
            self.logger.debug(f"Could not refresh chat view: {e}")
    
    def get_trainer(self):
        """Get the trainer instance."""
        return self.trainer
    
    def get_dataset(self):
        """Get the dataset instance."""
        return self.dataset
    
    def get_preprocessor(self):
        """Get the preprocessor instance."""
        return self.preprocessor
    
    def action_initialize_model(self):
        """Action to open model initialization screen."""
        self.push_screen(ModelInitializationScreen())
    
    def on_key(self, event):
        """Handle key presses."""
        if event.key == "ctrl+i":
            self.action_initialize_model()
    
    def on_unmount(self):
        """Clean up when app closes."""
        if self.trainer and self.trainer.is_training:
            self.trainer.stop_training()
        self.logger.info("Transformer TUI Application Closed")


# Run the app if this file is executed directly
if __name__ == "__main__":
    app = TransformerApp()
    app.run()
