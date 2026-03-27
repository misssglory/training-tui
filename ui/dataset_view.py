from textual.widgets import DataTable, Button, Input, Label, Static, ProgressBar
from textual.containers import Horizontal, Vertical
from textual import on
import threading
from pathlib import Path
import numpy as np


class DatasetView(Vertical):
    def compose(self):
        yield Label("Dataset Statistics", classes="title")
        yield DataTable(id="dataset_stats")
        yield Label("Vocabulary Statistics", classes="title")
        yield DataTable(id="vocab_stats")
        yield Label("Dataset Size Control", classes="title")
        with Horizontal():
            yield Input(placeholder="Max samples (None for all)", id="max_samples")
            yield Button("Apply Size Limit", id="apply_limit")
        yield Button("Load Sample Data", id="load_data")
        yield ProgressBar(id="loading_progress", total=100)
    
    def on_mount(self):
        stats_table = self.query_one("#dataset_stats")
        stats_table.add_columns("Split", "Samples", "Avg Length", "Tokens")
        
        vocab_table = self.query_one("#vocab_stats")
        vocab_table.add_columns("Split", "Total Tokens", "Unique Tokens", "Avg Length")
    
    @on(Button.Pressed, "#apply_limit")
    def apply_limit(self):
        max_samples_input = self.query_one("#max_samples")
        if max_samples_input.value:
            max_samples = int(max_samples_input.value) if max_samples_input.value != "None" else None
            self.app.config.set('data.max_samples', max_samples)
            self.app.notify(f"Dataset size limit set to {max_samples}")
    
    @on(Button.Pressed, "#load_data")
    def load_data(self):
        def load():
            # Sample data - replace with actual data loading
            texts = [
                "hello world", "how are you", "good morning", "good night",
                "what is your name", "my name is AI", "how old are you",
                "I am fine", "thank you", "you are welcome"
            ] * 1000  # Repeat to have more data
            
            labels = [
                "greeting", "question", "greeting", "greeting",
                "question", "statement", "question",
                "statement", "thanks", "response"
            ] * 1000
            
            # Load dataset
            max_samples = self.app.config.get('data.max_samples')
            self.app.dataset.load_data(texts, labels, max_samples)
            
            # Update statistics
            self._update_stats()
        
        threading.Thread(target=load, daemon=True).start()
    
    def _update_stats(self):
        stats_table = self.query_one("#dataset_stats")
        vocab_table = self.query_one("#vocab_stats")
        
        for split in ['train', 'val', 'test']:
            data = getattr(self.app.dataset, f"{split}_data")
            if data:
                X, y = data
                samples = len(X)
                avg_length = np.mean(np.sum(X != 0, axis=1))
                tokens = int(np.sum(X != 0))
                stats_table.add_row(split, str(samples), f"{avg_length:.2f}", str(tokens))
        
        if self.app.dataset.vocab_stats:
            for split, stats in self.app.dataset.vocab_stats.items():
                vocab_table.add_row(
                    split,
                    str(stats['total_tokens']),
                    str(stats['unique_tokens']),
                    f"{stats['avg_length']:.2f}"
                )
