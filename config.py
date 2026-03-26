import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

class Config:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.create_directories()
    
    def load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def save_config(self, config: Dict[str, Any]):
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        self.config = config
    
    def create_directories(self):
        for path_key, path in self.config["paths"].items():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
        self.save_config(self.config)
