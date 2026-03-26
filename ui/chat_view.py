from textual.widgets import Input, TextArea, Button, Label, Static, DataTable, TabbedContent, TabPane
from textual.containers import Horizontal, Vertical, Container
from textual import on
from textual.reactive import reactive
import threading
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import io
import tempfile
import climage
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Optional, Dict, Any
import keras

class ChatView(Vertical):
    """Chat view for interactive model testing with attention visualization."""
    
    def __init__(self):
        super().__init__()
        self.generation_thread = None
        self.is_generating = False
        self.attention_maps = []
        self.current_attention = None
    
    def compose(self):
        yield Label("Model Chat Interface", classes="title")
        
        with Horizontal():
            yield Label("Status: ", classes="status_label")
            yield Label("Ready", id="status", classes="status_value")
        
        with TabbedContent():
            with TabPane("Chat", id="chat_tab"):
                with Container(id="chat_container"):
                    yield TextArea(id="chat_history", disabled=True)
                    with Horizontal(id="input_container"):
                        yield Input(placeholder="Enter your message...", id="message_input")
                        yield Button("Send", id="send_button", variant="primary")
                        yield Button("Clear", id="clear_button", variant="default")
                    with Horizontal():
                        yield Button("Generate with Attention", id="generate_attention", variant="warning")
                        yield Button("Save Conversation", id="save_conversation", variant="default")
                        yield Label("Max tokens: ", classes="param_label")
                        yield Input(value="50", id="max_tokens", classes="param_input")
            
            with TabPane("Attention Visualization", id="attention_tab"):
                yield Static("Select a generated message to view attention maps", id="attention_info")
                with Horizontal():
                    yield Button("Refresh Attention", id="refresh_attention", variant="primary")
                    yield Button("Save Attention Map", id="save_attention", variant="default")
                    yield Button("Clear Attention", id="clear_attention", variant="error")
                yield Label("Attention Head Heatmaps", classes="subtitle")
                yield DataTable(id="attention_heads_table")
                yield Static(id="attention_preview", classes="attention_preview")
    
    def on_mount(self):
        """Initialize the chat view."""
        self.chat_history = self.query_one("#chat_history")
        self.message_input = self.query_one("#message_input")
        self.status_label = self.query_one("#status")
        self.attention_table = self.query_one("#attention_heads_table")
        self.attention_preview = self.query_one("#attention_preview")
        
        # Setup attention table
        self.attention_table.add_columns("Layer", "Head", "Min", "Max", "Mean")
        
        # Add welcome message
        self.add_message("System", "Welcome! I'm ready to chat. Type your message below to test the model.", "info")
        self.add_message("System", "Click 'Generate with Attention' to see attention heatmaps for the response.", "info")
        
        # Start periodic status check
        self.set_interval(1.0, self.update_status)
    
    def update_status(self):
        """Update model status indicator."""
        trainer = self.app.get_trainer()
        if trainer and trainer.is_training:
            self.status_label.update("Training in progress...")
            self.status_label.styles.background = "yellow"
            self.status_label.styles.color = "black"
        elif trainer and trainer.model:
            self.status_label.update("Model loaded - Ready")
            self.status_label.styles.background = "green"
            self.status_label.styles.color = "white"
        else:
            self.status_label.update("No model loaded")
            self.status_label.styles.background = "red"
            self.status_label.styles.color = "white"
    
    def add_message(self, sender: str, message: str, msg_type: str = "user"):
        """Add a message to the chat history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {sender}: {message}\n"
        
        current = self.chat_history.text
        self.chat_history.text = current + formatted_msg
        
        # Auto-scroll to bottom
        line_count = len(self.chat_history.text.splitlines())
        self.chat_history.cursor_location = (line_count, 0)
        
        # Store attention info if this is a model response
        if sender == "Model" and self.current_attention is not None:
            self.attention_maps.append({
                'timestamp': timestamp,
                'input': message,
                'attention': self.current_attention
            })
            self.current_attention = None
    
    @on(Button.Pressed, "#send_button")
    def send_message(self):
        """Send user message and get model response."""
        message = self.message_input.value.strip()
        if not message:
            self.app.notify("Please enter a message", severity="warning")
            return
        
        # Add user message to chat
        self.add_message("You", message)
        self.message_input.value = ""
        
        # Get model response
        self.generate_response(message, capture_attention=False)
    
    @on(Button.Pressed, "#generate_attention")
    def generate_with_attention(self):
        """Generate response and capture attention weights."""
        message = self.message_input.value.strip()
        if not message:
            self.app.notify("Please enter a message", severity="warning")
            return
        
        self.add_message("You", message)
        self.message_input.value = ""
        
        # Generate with attention capture
        self.generate_response(message, capture_attention=True)
    
    def generate_response(self, input_text: str, capture_attention: bool = False):
        """Generate model response in a separate thread."""
        trainer = self.app.get_trainer()
        if not trainer or not trainer.model:
            self.add_message("System", "Model not loaded. Please train or load a model first.", "error")
            return
        
        if self.is_generating:
            self.add_message("System", "Already generating a response. Please wait.", "warning")
            return
        
        max_tokens_input = self.query_one("#max_tokens")
        max_tokens = int(max_tokens_input.value) if max_tokens_input.value else 50
        
        def generate():
            self.is_generating = True
            self.call_from_thread(self.add_message, "Model", "Generating...", "info")
            
            try:
                start_time = time.time()
                
                if capture_attention:
                    # Generate with attention capture
                    response, attention_weights = self.generate_with_attention_capture(
                        trainer, input_text, max_tokens
                    )
                    self.current_attention = attention_weights
                    
                    # Update attention visualization
                    self.call_from_thread(self.display_attention_maps, attention_weights)
                else:
                    # Standard generation
                    response = trainer.generate_text(input_text, max_tokens)
                    self.current_attention = None
                
                generation_time = time.time() - start_time
                
                # Add response to chat
                self.call_from_thread(
                    self.add_message, 
                    "Model", 
                    f"{response}\n\n[Generated in {generation_time:.2f}s]",
                    "model"
                )
                
                self.call_from_thread(self.app.notify, f"Generated {len(response.split())} tokens in {generation_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                self.call_from_thread(self.add_message, "System", error_msg, "error")
                self.call_from_thread(self.app.notify, error_msg, severity="error")
            finally:
                self.is_generating = False
        
        self.generation_thread = threading.Thread(target=generate, daemon=True)
        self.generation_thread.start()
    
    def generate_with_attention_capture(self, trainer, input_text: str, max_tokens: int):
        """Generate text and capture attention weights."""
        preprocessor = self.app.get_preprocessor()
        
        # Prepare input
        sequence = preprocessor.text_to_sequence(input_text)
        input_seq = keras.ops.convert_to_tensor([sequence])
        
        # Generate output with attention capture
        output_seq = [preprocessor.word_index['<start>']]
        attention_weights = []
        
        for step in range(max_tokens):
            target_seq = keras.ops.convert_to_tensor([output_seq])
            
            # Forward pass
            predictions = trainer.model.predict([input_seq, target_seq], verbose=0)
            
            # Capture attention weights from decoder layers
            step_attention = []
            for layer in trainer.model.decoder_layers:
                if hasattr(layer.mha2, 'last_attention_weights') and layer.mha2.last_attention_weights is not None:
                    # Get attention for current step
                    attn = layer.mha2.last_attention_weights
                    step_attention.append(attn.numpy())
            
            attention_weights.append(step_attention)
            
            # Get predicted token
            predicted_id = int(keras.ops.argmax(predictions[0, -1, :]).numpy())
            
            if predicted_id == preprocessor.word_index['<end>']:
                break
            
            output_seq.append(predicted_id)
        
        # Convert to text
        response = preprocessor.sequence_to_text(output_seq[1:])
        
        # Process attention weights for visualization
        processed_attention = self.process_attention_weights(attention_weights, input_text, response)
        
        return response, processed_attention
    
    def process_attention_weights(self, attention_weights: List, input_text: str, response: str):
        """Process attention weights for visualization."""
        if not attention_weights:
            return None
        
        # Convert to numpy array
        processed = []
        for step_attn in attention_weights:
            step_processed = []
            for layer_attn in step_attn:
                if layer_attn is not None:
                    # Average over batch and heads
                    avg_attn = np.mean(layer_attn, axis=(0, 1))
                    step_processed.append(avg_attn)
            processed.append(step_processed)
        
        return {
            'attention_weights': processed,
            'input_text': input_text,
            'response': response,
            'num_layers': len(processed[0]) if processed else 0,
            'num_steps': len(processed)
        }
    
    def display_attention_maps(self, attention_data):
        """Display attention maps as heatmaps using climage."""
        if not attention_data or 'attention_weights' not in attention_data:
            self.attention_preview.update("No attention data available")
            return
        
        # Clear previous data
        self.attention_table.clear()
        
        # Get layer and head counts
        weights = attention_data['attention_weights']
        if not weights:
            return
        
        num_layers = len(weights[0]) if weights else 0
        num_heads = None
        
        # Add layer information to table
        for layer_idx in range(num_layers):
            # Collect attention for this layer across all steps
            layer_attentions = []
            for step in weights:
                if layer_idx < len(step):
                    layer_attentions.append(step[layer_idx])
            
            if layer_attentions:
                # Average across steps
                avg_attention = np.mean(layer_attentions, axis=0)
                
                if num_heads is None:
                    # Determine number of heads from attention shape
                    if len(avg_attention.shape) >= 2:
                        num_heads = avg_attention.shape[0]
                
                # Add row for each head
                if num_heads:
                    for head_idx in range(min(num_heads, 8)):  # Limit to first 8 heads
                        if head_idx < avg_attention.shape[0]:
                            head_attention = avg_attention[head_idx]
                            self.attention_table.add_row(
                                str(layer_idx + 1),
                                str(head_idx + 1),
                                f"{np.min(head_attention):.3f}",
                                f"{np.max(head_attention):.3f}",
                                f"{np.mean(head_attention):.3f}"
                            )
        
        # Generate heatmap for selected head (default: first layer, first head)
        self.show_attention_heatmap(attention_data, layer_idx=0, head_idx=0)
    
    def show_attention_heatmap(self, attention_data, layer_idx: int = 0, head_idx: int = 0):
        """Generate and display attention heatmap using climage."""
        weights = attention_data['attention_weights']
        if not weights:
            return
        
        # Extract attention for specific layer and head
        attentions = []
        for step in weights:
            if layer_idx < len(step):
                attn = step[layer_idx]
                if head_idx < attn.shape[0]:
                    attentions.append(attn[head_idx])
        
        if not attentions:
            self.attention_preview.update("No attention data for selected layer/head")
            return
        
        # Create heatmap
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Attention Heatmaps - Layer {layer_idx + 1}, Head {head_idx + 1}', fontsize=16)
        
        # Plot 1: Attention over time
        ax1 = axes[0, 0]
        attention_matrix = np.array(attentions)
        im1 = ax1.imshow(attention_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_title('Attention Weights Over Time')
        ax1.set_xlabel('Input Tokens')
        ax1.set_ylabel('Output Steps')
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Average attention per input token
        ax2 = axes[0, 1]
        avg_attention = np.mean(attention_matrix, axis=0)
        ax2.bar(range(len(avg_attention)), avg_attention)
        ax2.set_title('Average Attention per Input Token')
        ax2.set_xlabel('Token Index')
        ax2.set_ylabel('Attention Weight')
        
        # Plot 3: Attention distribution
        ax3 = axes[1, 0]
        ax3.hist(attention_matrix.flatten(), bins=50, edgecolor='black')
        ax3.set_title('Attention Weight Distribution')
        ax3.set_xlabel('Attention Weight')
        ax3.set_ylabel('Frequency')
        
        # Plot 4: Heatmap of last step
        ax4 = axes[1, 1]
        if len(attention_matrix) > 0:
            last_step = attention_matrix[-1]
            im4 = ax4.imshow([last_step], aspect='auto', cmap='viridis', interpolation='nearest')
            ax4.set_title('Last Step Attention')
            ax4.set_xlabel('Input Tokens')
            plt.colorbar(im4, ax=ax4)
        
        plt.tight_layout()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, dpi=100, bbox_inches='tight')
            tmp_path = tmp.name
        
        plt.close()
        
        # Convert to terminal image using climage
        try:
            with open(tmp_path, 'rb') as f:
                image_data = f.read()
            
            # Convert to terminal-compatible image
            output = climage.convert(
                image_data,
                width=80,
                is_unicode=True,
                is_256color=True,
                is_16color=False,
                palette='default'
            )
            
            # Display in terminal
            self.attention_preview.update(f"\n{output}\n\nAttention: Layer {layer_idx + 1}, Head {head_idx + 1}")
            
            # Clean up
            Path(tmp_path).unlink()
            
        except Exception as e:
            self.attention_preview.update(f"Error generating heatmap: {str(e)}")
            print(f"Heatmap generation error: {e}")
    
    @on(Button.Pressed, "#refresh_attention")
    def refresh_attention(self):
        """Refresh attention visualization."""
        if self.attention_maps:
            latest = self.attention_maps[-1]
            self.display_attention_maps(latest['attention'])
    
    @on(Button.Pressed, "#save_attention")
    def save_attention_map(self):
        """Save current attention map to file."""
        if not self.attention_maps:
            self.app.notify("No attention maps to save", severity="warning")
            return
        
        save_dir = Path(self.app.config.get('paths.attention_maps_dir'))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_path = save_dir / f"attention_map_{timestamp}.png"
        
        # Generate and save attention map
        latest = self.attention_maps[-1]
        attention_data = latest['attention']
        
        if attention_data and 'attention_weights' in attention_data:
            self.save_attention_to_file(attention_data, save_path)
            self.app.notify(f"Attention map saved to {save_path}")
    
    def save_attention_to_file(self, attention_data, save_path):
        """Save attention visualization to file."""
        weights = attention_data['attention_weights']
        if not weights:
            return
        
        # Create comprehensive attention visualization
        num_layers = len(weights[0]) if weights else 0
        num_heads = None
        
        fig, axes = plt.subplots(num_layers, 2, figsize=(15, 4 * num_layers))
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx in range(num_layers):
            # Collect attention for this layer
            layer_attentions = []
            for step in weights:
                if layer_idx < len(step):
                    layer_attentions.append(step[layer_idx])
            
            if layer_attentions:
                avg_attention = np.mean(layer_attentions, axis=0)
                
                if num_heads is None:
                    num_heads = avg_attention.shape[0]
                
                # Plot average attention across heads
                ax1 = axes[layer_idx, 0]
                avg_across_heads = np.mean(avg_attention, axis=0)
                ax1.imshow([avg_across_heads], aspect='auto', cmap='viridis')
                ax1.set_title(f'Layer {layer_idx + 1} - Average Across Heads')
                ax1.set_xlabel('Input Tokens')
                
                # Plot attention distribution
                ax2 = axes[layer_idx, 1]
                ax2.hist(avg_attention.flatten(), bins=50, edgecolor='black')
                ax2.set_title(f'Layer {layer_idx + 1} - Attention Distribution')
                ax2.set_xlabel('Attention Weight')
        
        plt.suptitle(f'Attention Maps - Input: "{attention_data["input_text"][:50]}..."')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @on(Button.Pressed, "#clear_attention")
    def clear_attention(self):
        """Clear attention visualization."""
        self.attention_table.clear()
        self.attention_preview.update("Attention visualization cleared")
        self.attention_maps = []
        self.current_attention = None
    
    @on(Button.Pressed, "#clear_button")
    def clear_chat(self):
        """Clear chat history."""
        self.chat_history.text = ""
        self.add_message("System", "Chat history cleared", "info")
    
    @on(Button.Pressed, "#save_conversation")
    def save_conversation(self):
        """Save chat conversation to file."""
        if not self.chat_history.text:
            self.app.notify("No conversation to save", severity="warning")
            return
        
        save_dir = Path(self.app.config.get('paths.output_dir')) / "conversations"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_path = save_dir / f"conversation_{timestamp}.txt"
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CONVERSATION LOG\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(self.chat_history.text)
        
        self.app.notify(f"Conversation saved to {save_path}")
    
    @on(DataTable.RowSelected, "#attention_heads_table")
    def on_attention_row_selected(self, event):
        """Handle attention head selection."""
        row = event.row
        if row is not None:
            # Get layer and head from selected row
            layer_idx = int(row[0]) - 1
            head_idx = int(row[1]) - 1
            
            if self.attention_maps:
                latest = self.attention_maps[-1]
                self.show_attention_heatmap(latest['attention'], layer_idx, head_idx)
