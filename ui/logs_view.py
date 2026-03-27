"""Logs view for displaying application logs."""
from textual.widgets import Select, RichLog, Button, Label, Input, TabbedContent, TabPane
from textual.containers import Horizontal, Vertical
from textual import on
from loguru import logger
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict, Any


class LogCapture:
    """Singleton class to capture logs globally."""
    _instance = None
    _logs = []
    _listeners = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup()
        return cls._instance
    
    def _setup(self):
        self._logs = []
        self._listeners = []
        
        # Add handler to capture logs
        logger.add(self._capture, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}")
    
    def _capture(self, message):
        """Capture log message."""
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'level': message.record['level'].name,
            'message': message.record['message'],
        }
        self._logs.append(log_entry)
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(log_entry)
            except Exception as e:
                print(f"Error in log listener: {e}")
    
    def get_logs(self, level: str = None, search: str = None, limit: int = None) -> List[Dict]:
        """Get logs with optional filtering."""
        logs = self._logs
        
        if level:
            logs = [l for l in logs if l['level'] == level.upper()]
        
        if search:
            logs = [l for l in logs if search.lower() in l['message'].lower()]
        
        if limit:
            logs = logs[-limit:]
        
        return logs
    
    def clear(self):
        """Clear all logs."""
        self._logs = []
    
    def add_listener(self, callback):
        """Add a listener for new logs."""
        self._listeners.append(callback)
    
    def remove_listener(self, callback):
        """Remove a listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)


class LogsView(Vertical):
    """View for displaying application logs."""
    
    DEFAULT_CSS = """
    LogsView {
        height: 100%;
        padding: 1;
    }
    
    #filter_bar {
        height: 3;
        margin: 1 0;
    }
    
    RichLog {
        height: 100%;
        border: solid $primary;
        margin: 1 0;
    }
    
    .label {
        margin: 0 1;
        content-align: center middle;
    }
    
    Select {
        margin: 0 1;
        width: 20;
    }
    
    Button {
        margin: 0 1;
    }
    
    #stats_label {
        padding: 1;
        background: $panel;
        margin: 1 0;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.log_capture = LogCapture()
        self.filter_level = "all"
        self.filter_search = ""
        self.auto_scroll = True
        self.test_logs_added = False
    
    def compose(self):
        """Create child widgets."""
        yield Label("Application Logs")
        
        # Filter bar
        with Horizontal(id="filter_bar"):
            yield Label("Level:")
            yield Select(
                [("All", "all"), ("INFO", "INFO"), ("WARNING", "WARNING"), ("ERROR", "ERROR")],
                id="level_filter",
                value="all",
                prompt="Select level"
            )
            
            yield Label("Search:")
            yield Input(placeholder="Search...", id="search_input")
            
            yield Button("Clear", id="clear_button", variant="default")
            yield Button("Copy", id="copy_button", variant="primary")
            yield Button("Export", id="export_button", variant="default")
        
        # Auto-scroll toggle
        with Horizontal():
            yield Label("Auto-scroll:")
            yield Select([("Yes", True), ("No", False)], id="auto_scroll", value=True)
        
        # Log display area
        yield RichLog(id="log_text", highlight=True, markup=True)
        
        # Stats
        yield Label("", id="stats_label")
    
    def on_mount(self):
        """Initialize the view."""
        self.update_logs()
        self.update_stats()
        
        # Add listener for new logs
        self.log_capture.add_listener(self.on_new_log)
        
        # Add test logs to show something
        if not self.test_logs_added:
            self.test_logs_added = True
            logger.info("Application started - Logs view initialized")
            logger.warning("This is a test warning message")
            logger.error("This is a test error message")
    
    def on_new_log(self, log_entry):
        """Handle new log entry."""
        self.call_from_thread_safe(self._add_log_to_display, log_entry)
    
    def _add_log_to_display(self, log_entry):
        """Add log to display."""
        # Check filters
        should_show = True
        if self.filter_level != "all" and log_entry['level'] != self.filter_level:
            should_show = False
        if self.filter_search and self.filter_search.lower() not in log_entry['message'].lower():
            should_show = False
        
        if should_show:
            log_widget = self.query_one("#log_text")
            formatted_log = self.format_log_entry(log_entry)
            log_widget.write(formatted_log)
            
            if self.auto_scroll:
                log_widget.scroll_end(animate=False)
        
        self.update_stats()
    
    def format_log_entry(self, log_entry: Dict) -> str:
        """Format a log entry for display."""
        level = log_entry['level']
        
        # Color coding
        if level == 'ERROR':
            color = 'red'
        elif level == 'WARNING':
            color = 'yellow'
        elif level == 'INFO':
            color = 'green'
        else:
            color = 'white'
        
        return f"[{color}]{log_entry['timestamp']} | {level:<8} | {log_entry['message']}[/{color}]"
    
    def update_logs(self):
        """Update logs display."""
        logs = self.log_capture.get_logs(
            level=None if self.filter_level == "all" else self.filter_level,
            search=self.filter_search if self.filter_search else None
        )
        
        log_widget = self.query_one("#log_text")
        log_widget.clear()
        for log in logs:
            log_widget.write(self.format_log_entry(log))
        
        if logs and self.auto_scroll:
            log_widget.scroll_end(animate=False)
    
    def update_stats(self):
        """Update statistics."""
        all_logs = self.log_capture.get_logs()
        errors = self.log_capture.get_logs(level="ERROR")
        warnings = self.log_capture.get_logs(level="WARNING")
        info = self.log_capture.get_logs(level="INFO")
        
        stats = f"Total: {len(all_logs)} | Errors: {len(errors)} | Warnings: {len(warnings)} | Info: {len(info)}"
        self.query_one("#stats_label").update(stats)
    
    @on(Select.Changed, "#level_filter")
    def on_level_filter(self, event):
        """Handle level filter change."""
        self.filter_level = event.value
        self.update_logs()
    
    @on(Input.Changed, "#search_input")
    def on_search_input(self, event):
        """Handle search input change."""
        self.filter_search = event.value
        self.update_logs()
    
    @on(Select.Changed, "#auto_scroll")
    def on_auto_scroll(self, event):
        """Handle auto-scroll toggle."""
        self.auto_scroll = event.value
    
    @on(Button.Pressed, "#clear_button")
    def on_clear(self):
        """Clear all logs."""
        self.log_capture.clear()
        self.update_logs()
        self.update_stats()
        self.app.notify("Logs cleared")
    
    @on(Button.Pressed, "#copy_button")
    def on_copy(self):
        """Copy logs to clipboard."""
        log_widget = self.query_one("#log_text")
        text = log_widget.text
        if text:
            try:
                import pyperclip
                pyperclip.copy(text)
                self.app.notify("Copied to clipboard!")
            except ImportError:
                self.app.notify("pip install pyperclip for copy support", severity="warning")
        else:
            self.app.notify("No logs to copy", severity="warning")
    
    @on(Button.Pressed, "#export_button")
    def on_export(self):
        """Export logs to file."""
        export_dir = Path.cwd() / "logs_exports"
        export_dir.mkdir(exist_ok=True)
        
        from time import strftime
        timestamp = strftime('%Y%m%d_%H%M%S')
        
        export_path = export_dir / f"logs_{timestamp}.txt"
        with open(export_path, 'w') as f:
            f.write(self.query_one("#log_text").text)
        
        self.app.notify(f"Logs exported to {export_path}")
    
    def on_unmount(self):
        """Remove listener."""
        self.log_capture.remove_listener(self.on_new_log)
