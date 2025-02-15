import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
from Model_124M_gpt2 import GPT, GPTConfig
import os

# Initialize model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding('gpt2')

# Load the trained model with efficient memory handling
def load_model():
    model = GPT(GPTConfig())
    
    # Check if we're running on HF Space
    if os.environ.get('SPACE_ID'):
        checkpoint_path = 'model/final_model.pt'
    else:
        checkpoint_path = 'final_model.pt'
    
    # Load model in chunks to handle memory efficiently
    checkpoint = torch.load(
        checkpoint_path,
        map_location=lambda storage, loc: storage
    )
    
    # Load state dict with memory efficiency
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if device == 'cuda':
        model = model.half()  # Use half precision on GPU
    
    model.to(device)
    model.eval()
    return model

# Load model with memory optimization
try:
    model = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def generate_text(prompt, max_length):
    if model is None:
        return "Error: Model failed to load. Please check the logs."
    
    # Ensure the model is in eval mode
    model.eval()
    
    # Encode the prompt - handle special tokens differently
    try:
        context = torch.tensor(
            enc.encode(prompt, disallowed_special=()), 
            dtype=torch.long, 
            device=device
        ).unsqueeze(0)
    except Exception as e:
        print(f"Encoding error: {e}")
        context = torch.tensor(
            enc.encode(prompt), 
            dtype=torch.long, 
            device=device
        ).unsqueeze(0)
    
    # Generate text
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            for _ in range(max_length):
                # Get model predictions
                logits, _ = model(context)
                logits = logits[:, -1, :]
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to context
                context = torch.cat([context, next_token], dim=1)
                
                # Optional: Stop if we generate an end of text token
                if next_token.item() == enc.encode("<|endoftext|>", disallowed_special=())[0]:
                    break
    
    # Decode the generated text
    try:
        generated_text = enc.decode(context[0].tolist())
    except Exception as e:
        print(f"Decoding error: {e}")
        # Fallback decoding
        generated_text = "Error in text generation. Please try again."
    
    return generated_text

# Create Gradio interface
def gradio_interface(prompt, sequence_length):
    try:
        sequence_length = int(sequence_length)
        generated = generate_text(prompt, sequence_length)
        return generated
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Define the Gradio app
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(
            lines=3, 
            placeholder="Enter your prompt here...",
            label="Prompt"
        ),
        gr.Slider(
            minimum=10,
            maximum=200,
            value=50,
            step=10,
            label="Sequence Length"
        )
    ],
    outputs=gr.Textbox(
        lines=5,
        label="Generated Text"
    ),
    title="GPT Text Generator",
    description="Enter a prompt and select the sequence length to generate text.",
    examples=[
        ["The quick brown fox", 50],
        ["Once upon a time", 100],
        ["In a galaxy far far away", 150]
    ]
)

# Launch the app
if __name__ == "__main__":
    iface.launch() 