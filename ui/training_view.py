"""Training view for model training."""
from textual.containers import Vertical, Horizontal
from textual.widgets import Button, Label, Input, ProgressBar
from textual import on
import threading
import time
from loguru import logger


class TrainingView(Vertical):
    """View for training configuration and monitoring."""
    
    DEFAULT_CSS = """
    TrainingView {
        height: 100%;
        padding: 1;
    }
    
    .config-section {
        border: solid $primary;
        margin: 1 0;
        padding: 1;
    }
    
    .progress-section {
        margin: 1 0;
        padding: 1;
    }
    
    .status-section {
        margin: 1 0;
        padding: 1;
        background: $panel;
    }
    
    .input-row {
        margin: 1 0;
    }
    
    .label {
        width: 15;
        content-align: right middle;
    }
    
    Input {
        width: 20;
        margin: 0 1;
    }
    
    Button {
        margin: 1 1;
    }
    
    ProgressBar {
        margin: 1 0;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.is_training = False
    
    def compose(self):
        """Create child widgets."""
        # Configuration section
        with Vertical(classes="config-section"):
            yield Label("Training Configuration")
            
            with Horizontal(classes="input-row"):
                yield Label("Epochs:", classes="label")
                yield Input(value="10", id="epochs_input")
            
            with Horizontal(classes="input-row"):
                yield Label("Batch Size:", classes="label")
                yield Input(value="32", id="batch_size_input")
            
            with Horizontal(classes="input-row"):
                yield Label("Learning Rate:", classes="label")
                yield Input(value="0.001", id="learning_rate_input")
        
        # Progress section
        with Vertical(classes="progress-section"):
            yield Label("Training Progress")
            yield ProgressBar(total=100, id="progress_bar")
            yield Label("Ready to start", id="progress_label")
        
        # Status section
        with Vertical(classes="status-section"):
            yield Label("Status: Not started", id="status_label")
            yield Label("Loss: --", id="loss_label")
            yield Label("Accuracy: --", id="accuracy_label")
        
        # Buttons
        with Horizontal():
            yield Button("Start Training", id="start_button", variant="primary")
            yield Button("Stop Training", id="stop_button", variant="error", disabled=True)
        
        # Add a test message
        logger.info("Training view initialized")
    
    @on(Button.Pressed, "#start_button")
    def on_start_training(self, event):
        """Start training."""
        if self.is_training:
            self.app.notify("Training already in progress", severity="warning")
            return
        
        try:
            epochs = int(self.query_one("#epochs_input").value)
            batch_size = int(self.query_one("#batch_size_input").value)
            learning_rate = float(self.query_one("#learning_rate_input").value)
            
            logger.info(f"Starting training with epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
            
            # Disable inputs
            self.query_one("#start_button").disabled = True
            self.query_one("#stop_button").disabled = False
            self.query_one("#epochs_input").disabled = True
            self.query_one("#batch_size_input").disabled = True
            self.query_one("#learning_rate_input").disabled = True
            
            self.is_training = True
            self.query_one("#status_label").update("Status: Training...")
            
            # Start training thread
            self.training_thread = threading.Thread(
                target=self._run_training,
                args=(epochs,),
                daemon=True
            )
            self.training_thread.start()
            
        except ValueError as e:
            self.app.notify(f"Invalid input: {e}", severity="error")
            logger.error(f"Invalid training input: {e}")
    
    def _run_training(self, epochs):
        """Run training simulation."""
        try:
            for epoch in range(epochs):
                if not self.is_training:
                    logger.info("Training stopped by user")
                    break
                
                # Calculate progress
                progress = ((epoch + 1) / epochs) * 100
                
                # Simulate training
                time.sleep(0.5)
                
                # Simulate metrics
                loss = 1.0 / (epoch + 1)
                accuracy = ((epoch + 1) / epochs) * 100
                
                # Update UI from background thread
                self.call_from_thread_safe(self._update_progress, progress, epoch + 1, epochs)
                self.call_from_thread_safe(self._update_metrics, loss, accuracy)
                
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            if self.is_training:
                self.call_from_thread_safe(self._training_complete)
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.call_from_thread_safe(self._training_error, str(e))
        finally:
            self.call_from_thread_safe(self._enable_ui)
    
    def _update_progress(self, progress, current_epoch, total_epochs):
        """Update progress bar."""
        self.query_one("#progress_bar").progress = progress
        self.query_one("#progress_label").update(f"Epoch {current_epoch}/{total_epochs}")
    
    def _update_metrics(self, loss, accuracy):
        """Update metrics display."""
        self.query_one("#loss_label").update(f"Loss: {loss:.4f}")
        self.query_one("#accuracy_label").update(f"Accuracy: {accuracy:.2f}%")
    
    def _training_complete(self):
        """Handle training completion."""
        self.query_one("#status_label").update("Status: Training Complete!")
        self.app.notify("Training completed successfully!")
        self.is_training = False
        logger.info("Training completed successfully")
    
    def _training_error(self, error):
        """Handle training error."""
        self.query_one("#status_label").update(f"Status: Error - {error}")
        self.app.notify(f"Training error: {error}", severity="error")
        self.is_training = False
        logger.error(f"Training error: {error}")
    
    def _enable_ui(self):
        """Re-enable UI after training."""
        self.query_one("#start_button").disabled = False
        self.query_one("#stop_button").disabled = True
        self.query_one("#epochs_input").disabled = False
        self.query_one("#batch_size_input").disabled = False
        self.query_one("#learning_rate_input").disabled = False
        
        if not self.is_training:
            self.query_one("#status_label").update("Status: Ready")
    
    @on(Button.Pressed, "#stop_button")
    def on_stop_training(self, event):
        """Stop training."""
        self.is_training = False
        self.query_one("#status_label").update("Status: Stopping...")
        self.app.notify("Stopping training...", severity="warning")
        logger.info("Training stop requested")
