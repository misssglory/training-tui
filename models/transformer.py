import keras
import keras.ops as K
from keras import layers
import numpy as np
from typing import Optional, Tuple, List, Union

class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(max_len, d_model)
    
    def positional_encoding(self, max_len, d_model):
        angle_rads = self.get_angles(np.arange(max_len)[:, np.newaxis],
                                      np.arange(d_model)[np.newaxis, :],
                                      d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return keras.ops.cast(pos_encoding, dtype=keras.backend.floatx())
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :keras.ops.shape(inputs)[1], :]

def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = keras.ops.matmul(q, k, transpose_b=True)
    dk = keras.ops.cast(keras.ops.shape(k)[-1], dtype=matmul_qk.dtype)
    scaled_attention_logits = matmul_qk / keras.ops.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = keras.ops.softmax(scaled_attention_logits, axis=-1)
    output = keras.ops.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        self.last_attention_weights = None
    
    def split_heads(self, x, batch_size):
        x = keras.ops.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return keras.ops.transpose(x, axes=(0, 2, 1, 3))
    
    def call(self, q, k, v, mask=None):
        batch_size = keras.ops.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        self.last_attention_weights = attention_weights
        scaled_attention = keras.ops.transpose(scaled_attention, axes=(0, 2, 1, 3))
        concat_attention = keras.ops.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output

def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask=None):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3

class Transformer(keras.Model):
    """Standard Transformer model for sequence-to-sequence tasks."""
    
    def __init__(self, vocab_size, max_len, d_model, num_heads, dff, num_layers, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.final_layer = layers.Dense(vocab_size)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        inp, tar = inputs
        enc_mask = self.create_padding_mask(inp)
        dec_mask = self.create_look_ahead_mask(tar)
        dec_padding_mask = self.create_padding_mask(inp)
        
        enc_output = self.embedding(inp)
        enc_output = self.pos_encoding(enc_output)
        enc_output = self.dropout(enc_output, training=training)
        
        for i, enc_layer in enumerate(self.encoder_layers):
            enc_output = enc_layer(enc_output, training, enc_mask)
        
        dec_output = self.embedding(tar)
        dec_output = self.pos_encoding(dec_output)
        dec_output = self.dropout(dec_output, training=training)
        
        for i, dec_layer in enumerate(self.decoder_layers):
            dec_output = dec_layer(
                dec_output, enc_output, training,
                dec_mask, dec_padding_mask
            )
        
        output = self.final_layer(dec_output)
        return output
    
    def create_padding_mask(self, seq):
        seq = keras.ops.cast(keras.ops.equal(seq, 0), dtype=keras.backend.floatx())
        return seq[:, keras.ops.newaxis, keras.ops.newaxis, :]
    
    def create_look_ahead_mask(self, size):
        mask = 1 - keras.ops.tril(keras.ops.ones((size, size)))
        return mask


class TransformerQA(keras.Model):
    """
    Transformer model specialized for Question Answering.
    
    This model can handle:
    1. Extractive QA: Find answer span in context
    2. Abstractive QA: Generate answer from context
    3. Multiple-choice QA: Select correct answer
    """
    
    def __init__(self, 
                 vocab_size: int,
                 max_len: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 num_layers: int,
                 dropout_rate: float = 0.1,
                 qa_type: str = "extractive"):  # extractive, abstractive, multiple_choice
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.qa_type = qa_type
        
        # Shared components
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.dropout = layers.Dropout(dropout_rate)
        
        # Encoder for context
        self.context_encoder = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Encoder for question
        self.question_encoder = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Cross-attention layers for combining context and question
        self.cross_attention = [
            MultiHeadAttention(d_model, num_heads)
            for _ in range(num_layers)
        ]
        
        # Task-specific output layers
        if qa_type == "extractive":
            # For extractive QA: predict start and end positions
            self.start_logits = layers.Dense(1)
            self.end_logits = layers.Dense(1)
            
        elif qa_type == "abstractive":
            # For abstractive QA: generate answer tokens
            self.decoder_layers = [
                DecoderLayer(d_model, num_heads, dff, dropout_rate)
                for _ in range(num_layers)
            ]
            self.final_layer = layers.Dense(vocab_size)
            
        elif qa_type == "multiple_choice":
            # For multiple-choice: classify answer options
            self.choice_classifier = layers.Dense(1, activation='sigmoid')
        
        # Attention storage for visualization
        self.last_attention_weights = []
    
    def call(self, inputs, training=False):
        """Forward pass based on QA type."""
        if self.qa_type == "extractive":
            return self.call_extractive(inputs, training)
        elif self.qa_type == "abstractive":
            return self.call_abstractive(inputs, training)
        elif self.qa_type == "multiple_choice":
            return self.call_multiple_choice(inputs, training)
        else:
            raise ValueError(f"Unknown QA type: {self.qa_type}")
    
    def call_extractive(self, inputs, training=False):
        """
        Extractive QA: Predict start and end positions of answer in context.
        
        Inputs: (context_ids, question_ids)
        Output: (start_logits, end_logits)
        """
        context_ids, question_ids = inputs
        
        # Encode context
        context_emb = self.embedding(context_ids)
        context_emb = self.pos_encoding(context_emb)
        context_emb = self.dropout(context_emb, training=training)
        
        context_mask = self.create_padding_mask(context_ids)
        
        for enc_layer in self.context_encoder:
            context_emb = enc_layer(context_emb, training, context_mask)
        
        # Encode question
        question_emb = self.embedding(question_ids)
        question_emb = self.pos_encoding(question_emb)
        question_emb = self.dropout(question_emb, training=training)
        
        question_mask = self.create_padding_mask(question_ids)
        
        for enc_layer in self.question_encoder:
            question_emb = enc_layer(question_emb, training, question_mask)
        
        # Cross-attend question to context
        combined_output = []
        attention_weights = []
        
        for i, cross_attn in enumerate(self.cross_attention):
            # Attend context to question
            attn_output = cross_attn(context_emb, question_emb, question_emb, question_mask)
            combined_output.append(attn_output)
            self.last_attention_weights.append(cross_attn.last_attention_weights)
        
        # Combine and project
        combined = keras.ops.add(combined_output)
        
        # Predict start and end positions
        start_logits = self.start_logits(combined)
        end_logits = self.end_logits(combined)
        
        return start_logits, end_logits
    
    def call_abstractive(self, inputs, training=False):
        """
        Abstractive QA: Generate answer tokens.
        
        Inputs: (context_ids, question_ids, answer_ids)
        Output: token logits
        """
        context_ids, question_ids, answer_ids = inputs
        
        # Encode context
        context_emb = self.embedding(context_ids)
        context_emb = self.pos_encoding(context_emb)
        context_emb = self.dropout(context_emb, training=training)
        
        context_mask = self.create_padding_mask(context_ids)
        
        for enc_layer in self.context_encoder:
            context_emb = enc_layer(context_emb, training, context_mask)
        
        # Encode question
        question_emb = self.embedding(question_ids)
        question_emb = self.pos_encoding(question_emb)
        question_emb = self.dropout(question_emb, training=training)
        
        question_mask = self.create_padding_mask(question_ids)
        
        for enc_layer in self.question_encoder:
            question_emb = enc_layer(question_emb, training, question_mask)
        
        # Combine context and question for decoding
        encoder_output = keras.ops.concatenate([context_emb, question_emb], axis=1)
        encoder_mask = keras.ops.concatenate([context_mask, question_mask], axis=1)
        
        # Decode answer
        answer_emb = self.embedding(answer_ids)
        answer_emb = self.pos_encoding(answer_emb)
        answer_emb = self.dropout(answer_emb, training=training)
        
        look_ahead_mask = self.create_look_ahead_mask(answer_ids)
        padding_mask = self.create_padding_mask(answer_ids)
        
        decoder_output = answer_emb
        for dec_layer in self.decoder_layers:
            decoder_output = dec_layer(
                decoder_output, encoder_output, training,
                look_ahead_mask, encoder_mask
            )
        
        logits = self.final_layer(decoder_output)
        return logits
    
    def call_multiple_choice(self, inputs, training=False):
        """
        Multiple-choice QA: Classify which answer is correct.
        
        Inputs: (context_ids, question_ids, answer_ids)
        Output: probability for each choice
        """
        context_ids, question_ids, answer_ids = inputs
        
        # Encode context
        context_emb = self.embedding(context_ids)
        context_emb = self.pos_encoding(context_emb)
        context_emb = self.dropout(context_emb, training=training)
        
        context_mask = self.create_padding_mask(context_ids)
        
        for enc_layer in self.context_encoder:
            context_emb = enc_layer(context_emb, training, context_mask)
        
        # Encode question
        question_emb = self.embedding(question_ids)
        question_emb = self.pos_encoding(question_emb)
        question_emb = self.dropout(question_emb, training=training)
        
        question_mask = self.create_padding_mask(question_ids)
        
        for enc_layer in self.question_encoder:
            question_emb = enc_layer(question_emb, training, question_mask)
        
        # Encode answer choice
        answer_emb = self.embedding(answer_ids)
        answer_emb = self.pos_encoding(answer_emb)
        answer_emb = self.dropout(answer_emb, training=training)
        
        answer_mask = self.create_padding_mask(answer_ids)
        
        # Combine all representations
        combined = keras.ops.concatenate([context_emb, question_emb, answer_emb], axis=1)
        
        # Global pooling
        pooled = keras.ops.mean(combined, axis=1)
        
        # Classify
        score = self.choice_classifier(pooled)
        return score
    
    def create_padding_mask(self, seq):
        """Create mask for padding tokens."""
        seq = keras.ops.cast(keras.ops.equal(seq, 0), dtype=keras.backend.floatx())
        return seq[:, keras.ops.newaxis, keras.ops.newaxis, :]
    
    def create_look_ahead_mask(self, seq):
        """Create look-ahead mask for autoregressive generation."""
        size = keras.ops.shape(seq)[1]
        mask = 1 - keras.ops.tril(keras.ops.ones((size, size)))
        return mask
    
    def get_attention_weights(self):
        """Retrieve stored attention weights for visualization."""
        return self.last_attention_weights
    
    def answer_question(self, context: str, question: str, preprocessor, max_answer_len: int = 50):
        """
        Answer a question using the trained model.
        
        Args:
            context: The context text
            question: The question to answer
            preprocessor: Text preprocessor for tokenization
            max_answer_len: Maximum answer length for abstractive QA
        
        Returns:
            Answer string and attention weights
        """
        # Tokenize inputs
        context_ids = preprocessor.text_to_sequence(context)
        question_ids = preprocessor.text_to_sequence(question)
        
        context_tensor = keras.ops.convert_to_tensor([context_ids])
        question_tensor = keras.ops.convert_to_tensor([question_ids])
        
        if self.qa_type == "extractive":
            # Get start and end positions
            start_logits, end_logits = self.predict([context_tensor, question_tensor], verbose=0)
            
            start_idx = int(keras.ops.argmax(start_logits[0]))
            end_idx = int(keras.ops.argmax(end_logits[0])) + 1
            
            # Extract answer span
            answer_tokens = context_ids[start_idx:end_idx]
            answer = preprocessor.sequence_to_text(answer_tokens)
            
            # Get attention weights
            attention_weights = self.get_attention_weights()
            
            return answer, attention_weights
        
        elif self.qa_type == "abstractive":
            # Generate answer token by token
            start_token = preprocessor.word_index['<start>']
            end_token = preprocessor.word_index['<end>']
            
            answer_ids = [start_token]
            
            for _ in range(max_answer_len):
                answer_tensor = keras.ops.convert_to_tensor([answer_ids])
                predictions = self.predict([context_tensor, question_tensor, answer_tensor], verbose=0)
                predicted_id = int(keras.ops.argmax(predictions[0, -1, :]))
                
                if predicted_id == end_token:
                    break
                
                answer_ids.append(predicted_id)
            
            answer = preprocessor.sequence_to_text(answer_ids[1:])
            attention_weights = self.get_attention_weights()
            
            return answer, attention_weights
        
        elif self.qa_type == "multiple_choice":
            # This would require multiple choice options
            # For simplicity, return a message
            return "Multiple choice QA requires answer options", None


class BertLikeTransformer(Transformer):
    """
    BERT-like encoder-only transformer for QA tasks.
    Uses [CLS] token for classification.
    """
    
    def __init__(self, vocab_size, max_len, d_model, num_heads, dff, num_layers, dropout_rate=0.1):
        super().__init__(vocab_size, max_len, d_model, num_heads, dff, num_layers, dropout_rate)
        
        # Override to use encoder only
        self.decoder_layers = None
        self.final_layer = layers.Dense(2)  # For start/end positions
        
    def call(self, inputs, training=False):
        """
        BERT-style forward pass.
        Input: [CLS] context [SEP] question [SEP]
        """
        input_ids = inputs
        
        # Create masks
        padding_mask = self.create_padding_mask(input_ids)
        
        # Embeddings
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Encoder layers
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, training, padding_mask)
        
        # Use [CLS] token (first token) for classification
        cls_output = x[:, 0, :]
        
        # Predict start and end positions
        logits = self.final_layer(cls_output)
        
        return logits
    
    def answer_question(self, context: str, question: str, preprocessor, max_answer_len: int = 50):
        """Answer using BERT-like model."""
        # Create input: [CLS] + context + [SEP] + question + [SEP]
        cls_token = preprocessor.word_index.get('<cls>', 0)
        sep_token = preprocessor.word_index.get('<sep>', 0)
        
        context_ids = preprocessor.text_to_sequence(context)
        question_ids = preprocessor.text_to_sequence(question)
        
        input_ids = [cls_token] + context_ids + [sep_token] + question_ids + [sep_token]
        
        # Pad to max_len
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        else:
            input_ids = input_ids + [0] * (self.max_len - len(input_ids))
        
        input_tensor = keras.ops.convert_to_tensor([input_ids])
        
        # Get predictions
        logits = self.predict(input_tensor, verbose=0)
        
        # For extractive QA, you'd need more complex span prediction
        # This is a simplified version
        answer = "BERT-style QA requires additional training"
        
        return answer, None


# Factory function to create QA models
def create_qa_model(model_type: str = "transformer_qa", 
                    vocab_size: int = 10000,
                    max_len: int = 100,
                    d_model: int = 512,
                    num_heads: int = 8,
                    dff: int = 2048,
                    num_layers: int = 6,
                    dropout_rate: float = 0.1,
                    qa_type: str = "extractive"):
    """
    Factory function to create various QA models.
    
    Args:
        model_type: "transformer", "transformer_qa", "bert_like"
        qa_type: For transformer_qa: "extractive", "abstractive", "multiple_choice"
    
    Returns:
        Configured model
    """
    if model_type == "transformer":
        return Transformer(vocab_size, max_len, d_model, num_heads, dff, num_layers, dropout_rate)
    
    elif model_type == "transformer_qa":
        return TransformerQA(vocab_size, max_len, d_model, num_heads, dff, num_layers, 
                            dropout_rate, qa_type)
    
    elif model_type == "bert_like":
        return BertLikeTransformer(vocab_size, max_len, d_model, num_heads, dff, num_layers, dropout_rate)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage in training
if __name__ == "__main__":
    # Test QA model creation
    vocab_size = 10000
    max_len = 100
    
    # Extractive QA model
    extractive_qa = create_qa_model(
        model_type="transformer_qa",
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=256,
        num_heads=4,
        dff=512,
        num_layers=4,
        qa_type="extractive"
    )
    
    print(f"Extractive QA model created with {extractive_qa.count_params():,} parameters")
    
    # Abstractive QA model
    abstractive_qa = create_qa_model(
        model_type="transformer_qa",
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=256,
        num_heads=4,
        dff=512,
        num_layers=4,
        qa_type="abstractive"
    )
    
    print(f"Abstractive QA model created with {abstractive_qa.count_params():,} parameters")
    
    # BERT-like model
    bert_qa = create_qa_model(
        model_type="bert_like",
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=256,
        num_heads=4,
        dff=512,
        num_layers=4
    )
    
    print(f"BERT-like QA model created with {bert_qa.count_params():,} parameters")
