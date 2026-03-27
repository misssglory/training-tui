from textual.widgets import Select, TextArea, Button, Label, Static, Input, TabbedContent, TabPane
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual import on
from textual.reactive import reactive
from loguru import logger
import threading
import queue
from datetime import datetime
from pathlib import Path
import json
import re
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
        self._log_queue = queue.Queue()
        
        # Add handler to capture logs
        logger.add(self._capture, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}")
    
    def _capture(self, message):
        """Capture log message."""
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'level': message.record['level'].name,
            'name': message.record['name'],
            'function': message.record['function'],
            'line': message.record['line'],
            'message': message.record['message'],
            'full': str(message)
        }
        self._logs.append(log_entry)
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(log_entry)
            except:
                pass
    
    def get_logs(self, level: str = None, search: str = None, limit: int = None) -> List[Dict]:
        """Get logs with optional filtering."""
        logs = self._logs
        
        if level:
            logs = [l for l in logs if l['level'] == level.upper()]
        
        if search:
            logs = [l for l in logs if search.lower() in l['message'].lower() or search.lower() in l['full'].lower()]
        
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
    """View for displaying application logs with filtering and copy capabilities."""
    
    DEFAULT_CSS = """
    LogsView {
        height: 100%;
    }
    
    #filter_bar {
        height: 5;
        margin: 1 0;
        padding: 0 1;
    }
    
    #log_text {
        height: 100%;
        border: solid $primary;
    }
    
    .log-level {
        margin: 0 1;
    }
    
    .filter-input {
        margin: 0 1;
        width: 100%;
    }
    
    Button {
        margin: 0 1;
    }
    
    #stats_label {
        padding: 1;
        background: $panel;
    }
    """

    
    def __init__(self):
        super().__init__()
        self.log_capture = LogCapture()
        self.current_logs = []
        self.filter_level = None
        self.filter_search = ""
        self.auto_scroll = True
    
    def compose(self):
        yield Label("Application Logs", classes="title")
        
        with Horizontal(id="filter_bar"):
            yield Label("Level:", classes="log-level")
            yield Select(
                [("All", "all"),
                 ("DEBUG", "DEBUG"),
                 ("INFO", "INFO"),
                 ("WARNING", "WARNING"),
                 ("ERROR", "ERROR"),
                 ("CRITICAL", "CRITICAL")],
                id="level_filter",
                value="all",
                classes="log-level"
            )
            
            yield Label("Search:", classes="log-level")
            yield Input(placeholder="Search logs...", id="search_input", classes="filter-input")
            
            yield Button("Clear", id="clear_button", variant="default")
            yield Button("Copy All", id="copy_button", variant="primary")
            yield Button("Copy Visible", id="copy_visible", variant="primary")
            yield Button("Export Logs", id="export_button", variant="default")
        
        with Horizontal():
            yield Label("Auto-scroll:", classes="log-level")
            # FIX: Use boolean values instead of strings
            yield Select(
                [("Yes", True), ("No", False)],  # Changed from ("true", "Yes") to use boolean values
                id="auto_scroll",
                value=True,  # Changed from "true" to True
                classes="log-level"
            )
            yield Label("", id="stats_label", classes="log-level")
        
        with TabbedContent():
            with TabPane("All Logs", id="all_logs"):
                yield TextArea(id="log_text", disabled=True)
            
            with TabPane("Errors Only", id="errors_tab"):
                yield TextArea(id="errors_text", disabled=True)
            
            with TabPane("Warnings", id="warnings_tab"):
                yield TextArea(id="warnings_text", disabled=True)
        
        # Add listener for new logs
        self.log_capture.add_listener(self.on_new_log)
    
    def on_mount(self):
        """Initialize the view."""
        self.update_all_logs()
        self.update_errors()
        self.update_warnings()
        self.update_stats()
    
    def on_new_log(self, log_entry):
        """Handle new log entry."""
        self.call_from_thread_safe(self._update_with_new_log, log_entry)
    
    def _update_with_new_log(self, log_entry):
        """Update UI with new log entry."""
        # Check if this log should be visible based on filters
        should_show = True
        
        if self.filter_level and self.filter_level != "all":
            if log_entry['level'] != self.filter_level:
                should_show = False
        
        if self.filter_search and self.filter_search not in log_entry['message'] and self.filter_search not in log_entry['full']:
            should_show = False
        
        if should_show:
            log_text = self.query_one("#log_text")
            if self.auto_scroll:
                current = log_text.text
                log_text.text = current + self.format_log_entry(log_entry)
                # Auto-scroll to bottom
                log_text.cursor_location = (len(log_text.text.splitlines()), 0)
            else:
                current = log_text.text
                log_text.text = current + self.format_log_entry(log_entry)
        
        # Update other tabs
        if log_entry['level'] == 'ERROR':
            errors_text = self.query_one("#errors_text")
            errors_text.text = errors_text.text + self.format_log_entry(log_entry)
        elif log_entry['level'] == 'WARNING':
            warnings_text = self.query_one("#warnings_text")
            warnings_text.text = warnings_text.text + self.format_log_entry(log_entry)
        
        self.update_stats()
    
    def format_log_entry(self, log_entry: Dict) -> str:
        """Format a log entry for display."""
        # Color coding based on level
        level = log_entry['level']
        if level == 'ERROR':
            color = 'red'
        elif level == 'WARNING':
            color = 'yellow'
        elif level == 'INFO':
            color = 'green'
        else:
            color = 'white'
        
        # Format with ANSI colors for better readability
        return f"{log_entry['timestamp']} | \033[{self._get_color_code(color)}m{level:<8}\033[0m | {log_entry['name']}:{log_entry['function']}:{log_entry['line']} | {log_entry['message']}\n"
    
    def _get_color_code(self, color: str) -> str:
        """Get ANSI color code."""
        colors = {
            'red': '31',
            'green': '32',
            'yellow': '33',
            'blue': '34',
            'magenta': '35',
            'cyan': '36',
            'white': '37'
        }
        return colors.get(color, '37')
    
    def update_all_logs(self):
        """Update all logs view with current filters."""
        logs = self.log_capture.get_logs(
            level=None if self.filter_level == "all" else self.filter_level,
            search=self.filter_search if self.filter_search else None
        )
        
        log_text = "\n".join([self.format_log_entry(log) for log in logs])
        self.query_one("#log_text").text = log_text
    
    def update_errors(self):
        """Update errors only view."""
        errors = self.log_capture.get_logs(level="ERROR")
        error_text = "\n".join([self.format_log_entry(log) for log in errors])
        self.query_one("#errors_text").text = error_text
    
    def update_warnings(self):
        """Update warnings only view."""
        warnings = self.log_capture.get_logs(level="WARNING")
        warning_text = "\n".join([self.format_log_entry(log) for log in warnings])
        self.query_one("#warnings_text").text = warning_text
    
    def update_stats(self):
        """Update statistics display."""
        all_logs = self.log_capture.get_logs()
        errors = self.log_capture.get_logs(level="ERROR")
        warnings = self.log_capture.get_logs(level="WARNING")
        info = self.log_capture.get_logs(level="INFO")
        
        stats = f"Total: {len(all_logs)} | Errors: {len(errors)} | Warnings: {len(warnings)} | Info: {len(info)}"
        self.query_one("#stats_label").update(stats)
    
    def copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        try:
            import pyperclip
            pyperclip.copy(text)
            self.app.notify("Copied to clipboard!")
        except ImportError:
            # Fallback for systems without pyperclip
            self.app.notify("Clipboard copy requires pyperclip: pip install pyperclip", severity="warning")
    
    @on(Select.Changed, "#level_filter")
    def on_level_filter(self, event):
        """Handle level filter change."""
        self.filter_level = event.value if event.value != "all" else None
        self.update_all_logs()
    
    @on(Input.Changed, "#search_input")
    def on_search_input(self, event):
        """Handle search input change."""
        self.filter_search = event.value
        self.update_all_logs()
    
    @on(Select.Changed, "#auto_scroll")
    def on_auto_scroll(self, event):
        """Handle auto-scroll toggle."""
        # FIX: Handle boolean value instead of string
        self.auto_scroll = event.value  # Now event.value will be True or False
    
    @on(Button.Pressed, "#clear_button")
    def on_clear(self):
        """Clear all logs."""
        self.log_capture.clear()
        self.update_all_logs()
        self.update_errors()
        self.update_warnings()
        self.update_stats()
        self.app.notify("Logs cleared")
    
    @on(Button.Pressed, "#copy_button")
    def on_copy_all(self):
        """Copy all logs to clipboard."""
        text = self.query_one("#log_text").text
        if text:
            self.copy_to_clipboard(text)
        else:
            self.app.notify("No logs to copy", severity="warning")
    
    @on(Button.Pressed, "#copy_visible")
    def on_copy_visible(self):
        """Copy visible logs to clipboard."""
        text = self.query_one("#log_text").text
        if text:
            self.copy_to_clipboard(text)
        else:
            self.app.notify("No logs to copy", severity="warning")
    
    @on(Button.Pressed, "#export_button")
    def on_export(self):
        """Export logs to file."""
        export_dir = Path(self.app.config.get('paths.logs_dir'))
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = __import__('time').strftime('%Y%m%d_%H%M%S')
        
        # Export all logs
        all_path = export_dir / f"all_logs_{timestamp}.txt"
        with open(all_path, 'w') as f:
            f.write(self.query_one("#log_text").text)
        
        # Export errors
        errors_path = export_dir / f"errors_{timestamp}.txt"
        with open(errors_path, 'w') as f:
            f.write(self.query_one("#errors_text").text)
        
        # Export warnings
        warnings_path = export_dir / f"warnings_{timestamp}.txt"
        with open(warnings_path, 'w') as f:
            f.write(self.query_one("#warnings_text").text)
        
        # Export as JSON
        json_path = export_dir / f"logs_{timestamp}.json"
        all_logs = self.log_capture.get_logs()
        with open(json_path, 'w') as f:
            json.dump(all_logs, f, indent=2, default=str)
        
        self.app.notify(f"Logs exported to {export_dir}")
    
    def on_unmount(self):
        """Remove listener when unmounting."""
        self.log_capture.remove_listener(self.on_new_log)


# Add logging handlers to capture errors and warnings
def capture_exception(exc_type, exc_value, exc_traceback):
    """Capture unhandled exceptions."""
    logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("Unhandled exception")


# Install exception hook
import sys
sys.excepthook = capture_exception
