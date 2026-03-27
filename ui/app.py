"""Main application."""
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane
from ui.logs_view import LogsView
from ui.training_view import TrainingView
from loguru import logger


class TransformerApp(App):
    """Main transformer TUI application."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    TabbedContent {
        height: 100%;
    }
    
    TabPane {
        height: 100%;
        padding: 0;
    }
    
    .title {
        text-style: bold;
        padding: 1;
        background: $primary;
        color: $text;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.config = {}
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        
        with TabbedContent():
            with TabPane("Training", id="training_tab"):
                yield TrainingView()
            
            with TabPane("Logs", id="logs_tab"):
                yield LogsView()
        
        yield Footer()
    
    def on_mount(self):
        """Handle app mount."""
        self.title = "Transformer TUI"
        self.sub_title = "Model Training Interface"
        logger.info("Application started")
    
    def action_quit(self):
        """Quit the application."""
        logger.info("Quitting application")
        self.exit()


def run_app():
    """Run the application."""
    app = TransformerApp()
    app.run()


if __name__ == "__main__":
    run_app()
