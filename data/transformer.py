import keras
import keras.ops as K
from keras import layers
import numpy as np

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
    def __init__(self, vocab_size, max_len, d_model, num_heads, dff, num_layers, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
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
