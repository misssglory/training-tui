import numpy as np
from typing import List, Tuple, Optional
import keras
from pathlib import Path
import pickle

class TextDataset:
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.vocab_stats = None
    
    def load_data(self, texts: List[str], labels: List[str], max_samples: Optional[int] = None):
        if max_samples and max_samples < len(texts):
            texts = texts[:max_samples]
            labels = labels[:max_samples]
        
        # Fit preprocessor if not already fitted
        if not self.preprocessor.vocab_created:
            self.preprocessor.fit(texts + labels)
        
        # Prepare dataset
        X, y = self.preprocessor.prepare_dataset(texts, labels)
        
        # Split data
        train_split = self.config.get('data.train_split', 0.8)
        val_split = self.config.get('data.val_split', 0.1)
        
        n = len(X)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)
        
        self.train_data = (X[:train_end], y[:train_end])
        self.val_data = (X[train_end:val_end], y[train_end:val_end])
        self.test_data = (X[val_end:], y[val_end:])
        
        # Calculate vocabulary statistics
        self.vocab_stats = self._get_vocab_stats()
        
        return self
    
    def _get_vocab_stats(self):
        stats = {}
        for name, (X, y) in [('train', self.train_data), ('val', self.val_data), ('test', self.test_data)]:
            if X is not None:
                stats[name] = {
                    'total_tokens': int(np.sum(X != 0)),
                    'unique_tokens': len(np.unique(X)),
                    'avg_length': float(np.mean(np.sum(X != 0, axis=1)))
                }
        return stats
    
    def get_batch_generator(self, data, batch_size, training=True):
        X, y = data
        dataset = keras.utils.timeseries_dataset_from_array(
            X, y, sequence_length=1, batch_size=batch_size, shuffle=training
        )
        return dataset
    
    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({
                'train_data': self.train_data,
                'val_data': self.val_data,
                'test_data': self.test_data,
                'vocab_stats': self.vocab_stats
            }, f)
    
    def load(self, path: Path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.train_data = data['train_data']
            self.val_data = data['val_data']
            self.test_data = data['test_data']
            self.vocab_stats = data['vocab_stats']
