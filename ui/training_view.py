from textual.widgets import Button, TextArea, Label, Static, ProgressBar, DataTable
from textual.containers import Horizontal, Vertical
from textual import on
import threading
from loguru import logger
import json
from pathlib import Path
import time
from datetime import datetime

class TrainingView(Vertical):
    """Training view with logs, history, and progress monitoring."""
    
    def compose(self):
        yield Label("Training Control", classes="title")
        with Horizontal():
            yield Button("Start Training", id="start_training", variant="primary")
            yield Button("Stop Training", id="stop_training", variant="error")
            yield Button("Save Model", id="save_model", variant="success")
            yield Button("Load Best Model", id="load_best", variant="warning")
        
        yield Label("Training Progress", classes="title")
        with Horizontal():
            yield ProgressBar(id="training_progress", total=100)
            yield Label("0%", id="progress_percent")
        
        yield Label("Training Metrics", classes="title")
        yield DataTable(id="metrics_table")
        
        yield Label("Training Logs", classes="title")
        yield TextArea(id="training_logs", disabled=True)
        
        yield Label("Training History", classes="title")
        yield TextArea(id="training_history", disabled=True)
        
        yield Label("Latest Checkpoint", classes="title")
        yield Static(id="checkpoint_info")
    
    def on_mount(self):
        """Initialize view components."""
        self.logs_text = self.query_one("#training_logs")
        self.history_text = self.query_one("#training_history")
        self.metrics_table = self.query_one("#metrics_table")
        self.progress_bar = self.query_one("#training_progress")
        self.progress_percent = self.query_one("#progress_percent")
        self.checkpoint_info = self.query_one("#checkpoint_info")
        
        # Setup metrics table
        self.metrics_table.add_columns("Epoch", "Loss", "Accuracy", "Val Loss", "Val Accuracy", "Time")
        
        # Add log handler
        logger.add(self._log_callback)
        
        # Start periodic checkpoint check
        self.set_interval(2.0, self._check_checkpoints)
    
    def _log_callback(self, message):
        """Callback for loguru messages."""
        self.call_from_thread(self._update_logs, str(message))
    
    def _update_logs(self, message):
        """Update logs text area."""
        current = self.logs_text.text
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        self.logs_text.text = current + formatted_msg
        # Auto-scroll to bottom
        self.logs_text.cursor_location = (len(self.logs_text.text.splitlines()), 0)
    
    def _update_history(self, epoch_data):
        """Update training history."""
        current = self.history_text.text
        history_entry = json.dumps(epoch_data, indent=2)
        self.history_text.text = current + history_entry + "\n" + "-" * 50 + "\n"
        
        # Update metrics table
        self.metrics_table.add_row(
            str(epoch_data['epoch']),
            f"{epoch_data['loss']:.4f}",
            f"{epoch_data['accuracy']:.4f}",
            f"{epoch_data['val_loss']:.4f}",
            f"{epoch_data['val_accuracy']:.4f}",
            epoch_data.get('time', 'N/A')
        )
        
        # Update progress
        trainer = self.app.get_trainer()
        if trainer:
            total_epochs = self.app.config.get('model.epochs', 100)
            progress = (epoch_data['epoch'] / total_epochs) * 100
            self.progress_bar.progress = progress
            self.progress_percent.update(f"{progress:.1f}%")
    
    def _check_checkpoints(self):
        """Check for new checkpoints."""
        trainer = self.app.get_trainer()
        if trainer:
            best_model = trainer.find_best_model()
            if best_model:
                self.checkpoint_info.update(f"Latest best model: {best_model.name}")
    
    @on(Button.Pressed, "#start_training")
    def start_training(self):
        """Start training in background thread."""
        trainer = self.app.get_trainer()
        
        if not trainer:
            self.app.notify("Trainer not initialized!", severity="error")
            return
        
        if trainer.is_training:
            self.app.notify("Training already in progress!", severity="warning")
            return
        
        # Clear previous logs and history
        self.logs_text.text = ""
        self.history_text.text = ""
        self.metrics_table.clear()
        
        # Start training
        logger.info("Starting training...")
        self.app.notify("Training started in background")
        
        trainer.start_training(on_epoch_end=self._update_history)
    
    @on(Button.Pressed, "#stop_training")
    def stop_training(self):
        """Stop training."""
        trainer = self.app.get_trainer()
        if trainer and trainer.is_training:
            logger.info("Stopping training...")
            trainer.stop_training()
            self.app.notify("Training stopped")
    
    @on(Button.Pressed, "#save_model")
    def save_model(self):
        """Save current model."""
        trainer = self.app.get_trainer()
        if trainer and trainer.model:
            models_dir = Path(self.app.config.get('paths.models_dir'))
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            model_path = models_dir / f'model_{timestamp}.keras'
            
            try:
                trainer.model.save(model_path)
                logger.info(f"Model saved to {model_path}")
                self.app.notify(f"Model saved: {model_path.name}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
                self.app.notify(f"Error saving model: {e}", severity="error")
    
    @on(Button.Pressed, "#load_best")
    def load_best_model(self):
        """Load the best saved model."""
        trainer = self.app.get_trainer()
        if trainer:
            best_model_path = trainer.find_best_model()
            if best_model_path:
                try:
                    trainer.model.load_weights(best_model_path)
                    logger.info(f"Loaded best model from {best_model_path}")
                    self.app.notify(f"Loaded best model: {best_model_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    self.app.notify(f"Error loading model: {e}", severity="error")
            else:
                self.app.notify("No saved models found", severity="warning")
