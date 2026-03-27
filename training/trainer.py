import keras
import numpy as np
from pathlib import Path
import time
from typing import Dict, Any, Optional
import threading
import queue
from loguru import logger
import json

class Trainer:
    def __init__(self, config, model, dataset, preprocessor):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.history = []
        self.is_training = False
        self.training_thread = None
        self.log_queue = queue.Queue()
        
        # Setup precision
        precision = config.get('model.precision', 'fp32')
        if precision == 'fp16':
            keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Build model if not already built (with dummy input)
        if not model.built:
            self._build_model_with_dummy_input()
        
        # Compile model
        self._compile_model()

    def _build_model_with_dummy_input(self):
        """Build the model with dummy input."""
        try:
            batch_size = self.config.get('model.batch_size', 32)
            max_len = self.config.get('model.max_len', 100)
            
            # Create dummy inputs
            dummy_input = keras.ops.zeros((batch_size, max_len), dtype='int32')
            dummy_target = keras.ops.zeros((batch_size, max_len), dtype='int32')
            
            # Call the model to build it
            if isinstance(self.model, (Transformer, TransformerQA, BertLikeTransformer)):
                self.model([dummy_input, dummy_target], training=False)
            elif hasattr(self.model, 'call'):
                try:
                    self.model(dummy_input, training=False)
                except:
                    pass
            
            logger.info(f"Model built successfully with input shape: ({batch_size}, {max_len})")
            logger.info(f"Model parameters: {self.model.count_params():,}")
            
        except Exception as e:
            logger.warning(f"Could not build model automatically: {e}")
            logger.info("Model will be built during first training step") 


    def _compile_model(self):
        learning_rate = self.config.get('model.learning_rate', 0.001)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
    
    def find_best_model(self) -> Optional[Path]:
        models_dir = Path(self.config.get('paths.models_dir'))
        if models_dir.exists():
            checkpoints = list(models_dir.glob('best_model_*.keras'))
            if checkpoints:
                return max(checkpoints, key=lambda p: p.stat().st_mtime)
        return None
    
    def train(self, on_epoch_end=None):
        self.is_training = True
        best_model_path = self.find_best_model()
        
        if best_model_path:
            logger.info(f"Loading best model from {best_model_path}")
            self.model.load_weights(best_model_path)
        
        train_data = self.dataset.get_batch_generator(
            self.dataset.train_data, 
            self.config.get('model.batch_size', 32),
            training=True
        )
        val_data = self.dataset.get_batch_generator(
            self.dataset.val_data,
            self.config.get('model.batch_size', 32),
            training=False
        )
        
        epochs = self.config.get('model.epochs', 100)
        
        callbacks = self._create_callbacks()
        
        try:
            history = self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            for epoch in range(len(history.history['loss'])):
                epoch_data = {
                    'epoch': epoch + 1,
                    'loss': history.history['loss'][epoch],
                    'accuracy': history.history['accuracy'][epoch],
                    'val_loss': history.history['val_loss'][epoch],
                    'val_accuracy': history.history['val_accuracy'][epoch]
                }
                self.history.append(epoch_data)
                
                if on_epoch_end:
                    on_epoch_end(epoch_data)
                
                logger.info(f"Epoch {epoch+1}: loss={epoch_data['loss']:.4f}, "
                           f"acc={epoch_data['accuracy']:.4f}, "
                           f"val_loss={epoch_data['val_loss']:.4f}")
        
        except Exception as e:
            logger.error(f"Training error: {e}")
        
        finally:
            self.is_training = False
            self._save_history()
    
    def _create_callbacks(self):
        models_dir = Path(self.config.get('paths.models_dir'))
        checkpoint = keras.callbacks.ModelCheckpoint(
            models_dir / 'best_model_{epoch:02d}_{val_loss:.4f}.keras',
            monitor=self.config.get('training.checkpoint_monitor', 'val_loss'),
            mode=self.config.get('training.checkpoint_mode', 'min'),
            save_best_only=True,
            save_weights_only=False
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=self.config.get('training.checkpoint_monitor', 'val_loss'),
            patience=self.config.get('training.early_stopping_patience', 10),
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=self.config.get('training.checkpoint_monitor', 'val_loss'),
            factor=self.config.get('training.reduce_lr_factor', 0.5),
            patience=self.config.get('training.reduce_lr_patience', 5)
        )
        
        return [checkpoint, early_stopping, reduce_lr]
    
    def _save_history(self):
        history_dir = Path(self.config.get('paths.history_dir'))
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        history_path = history_dir / f'training_history_{timestamp}.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        logger.info(f"Training history saved to {history_path}")
    
    def start_training(self, on_epoch_end=None):
        self.training_thread = threading.Thread(
            target=self.train,
            args=(on_epoch_end,),
            daemon=True
        )
        self.training_thread.start()
    
    def stop_training(self):
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
    
    def generate_text(self, input_text: str, max_length: int = 50) -> str:
        # Prepare input
        sequence = self.preprocessor.text_to_sequence(input_text)
        input_seq = keras.ops.convert_to_tensor([sequence])
        
        # Generate output
        output_seq = [self.preprocessor.word_index['<start>']]
        
        for _ in range(max_length):
            target_seq = keras.ops.convert_to_tensor([output_seq])
            predictions = self.model.predict([input_seq, target_seq], verbose=0)
            predicted_id = keras.ops.argmax(predictions[0, -1, :]).numpy()
            
            if predicted_id == self.preprocessor.word_index['<end>']:
                break
            
            output_seq.append(predicted_id)
        
        return self.preprocessor.sequence_to_text(output_seq[1:])
    
    def get_attention_weights(self, input_text: str) -> np.ndarray:
        sequence = self.preprocessor.text_to_sequence(input_text)
        input_seq = keras.ops.convert_to_tensor([sequence])
        
        # Get attention weights from the first decoder layer
        attention_weights = []
        for layer in self.model.decoder_layers:
            attention_weights.append(layer.mha2.last_attention_weights)
        
        return attention_weights
