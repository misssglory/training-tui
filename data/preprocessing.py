import numpy as np
from collections import Counter
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import keras

class TextPreprocessor:
    def __init__(self, vocab_size=10000, max_len=100):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word_index = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.index_word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.word_counts = Counter()
        self.vocab_created = False
    
    def fit(self, texts: List[str]):
        for text in texts:
            words = text.lower().split()
            self.word_counts.update(words)
        
        # Create vocabulary
        most_common = self.word_counts.most_common(self.vocab_size - 4)
        for word, count in most_common:
            idx = len(self.word_index)
            self.word_index[word] = idx
            self.index_word[idx] = word
        
        self.vocab_created = True
    
    def text_to_sequence(self, text: str) -> List[int]:
        if not self.vocab_created:
            raise ValueError("Preprocessor not fitted yet!")
        
        words = text.lower().split()
        sequence = [self.word_index.get(word, self.word_index['<unk>']) for word in words]
        return sequence[:self.max_len]
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        words = [self.index_word.get(idx, '<unk>') for idx in sequence]
        return ' '.join(words)
    
    def prepare_dataset(self, texts: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        sequences = [self.text_to_sequence(text) for text in texts]
        label_sequences = [self.text_to_sequence(label) for label in labels]
        
        # Pad sequences
        padded_sequences = keras.utils.pad_sequences(
            sequences, maxlen=self.max_len, padding='post', value=0
        )
        padded_labels = keras.utils.pad_sequences(
            label_sequences, maxlen=self.max_len, padding='post', value=0
        )
        
        return padded_sequences, padded_labels
    
    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({
                'word_index': self.word_index,
                'index_word': self.index_word,
                'word_counts': self.word_counts,
                'vocab_size': self.vocab_size,
                'max_len': self.max_len,
                'vocab_created': self.vocab_created
            }, f)
    
    def load(self, path: Path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word_index = data['word_index']
            self.index_word = data['index_word']
            self.word_counts = data['word_counts']
            self.vocab_size = data['vocab_size']
            self.max_len = data['max_len']
            self.vocab_created = data['vocab_created']
