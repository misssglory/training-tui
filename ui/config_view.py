from textual.widgets import DataTable, Button, Input, Label, Select
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual import on
import json

class ConfigView(Vertical):
    def compose(self):
        yield Label("Model Configuration", classes="title")
        yield DataTable(id="config_table")
        with Horizontal():
            yield Button("Load Config", id="load_config")
            yield Button("Save Config", id="save_config")
            yield Button("Reset", id="reset_config")
    
    def on_mount(self):
        table = self.query_one("#config_table")
        table.add_columns("Parameter", "Value")
        
        # Load current config
        config = self.app.config.config
        self._populate_table(config)
    
    def _populate_table(self, config, prefix=""):
        table = self.query_one("#config_table")
        table.clear()
        
        for key, value in config.items():
            if isinstance(value, dict):
                self._populate_table(value, f"{prefix}{key}.")
            else:
                table.add_row(f"{prefix}{key}", str(value))
    
    @on(Button.Pressed, "#load_config")
    def load_config(self):
        # Open file picker
        from textual.widgets import FileInput
        self.app.push_screen("file_picker")
    
    @on(Button.Pressed, "#save_config")
    def save_config(self):
        # Save current config
        self.app.config.save_config(self.app.config.config)
        self.app.notify("Config saved successfully!")
    
    @on(Button.Pressed, "#reset_config")
    def reset_config(self):
        # Reset to default
        with open("config.json", 'r') as f:
            default_config = json.load(f)
        self.app.config.config = default_config
        self._populate_table(default_config)
        self.app.notify("Config reset to default!")
