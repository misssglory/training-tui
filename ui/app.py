from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, TabbedContent, TabPane
from ui.config_view import ConfigView
from ui.dataset_view import DatasetView
from ui.training_view import TrainingView
from ui.model_summary_view import ModelSummaryView
from ui.chat_view import ChatView
from config import Config
from models.transformer import Transformer
from data.preprocessing import TextPreprocessor
from data.dataset import TextDataset
from training.trainer import Trainer
from utils.logger import setup_logger

class TransformerApp(App):
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
    """
    
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.logger = setup_logger(self.config)
        self.preprocessor = TextPreprocessor(
            vocab_size=self.config.get('model.vocab_size', 10000),
            max_len=self.config.get('model.max_len', 100)
        )
        self.dataset = TextDataset(self.config, self.preprocessor)
        self.model = None
        self.trainer = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Configuration", id="config"):
                yield ConfigView()
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
        self._create_model()
    
    def _create_model(self):
        self.model = Transformer(
            vocab_size=self.config.get('model.vocab_size', 10000),
            max_len=self.config.get('model.max_len', 100),
            d_model=self.config.get('model.d_model', 512),
            num_heads=self.config.get('model.num_heads', 8),
            dff=self.config.get('model.dff', 2048),
            num_layers=self.config.get('model.num_layers', 6),
            dropout_rate=self.config.get('model.dropout_rate', 0.1)
        )
        
        self.trainer = Trainer(self.config, self.model, self.dataset, self.preprocessor)
    
    def get_trainer(self):
        return self.trainer
    
    def get_dataset(self):
        return self.dataset
    
    def get_preprocessor(self):
        return self.preprocessor

if __name__ == "__main__":
    app = TransformerApp()
    app.run()
