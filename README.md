# GPT Text Generator
A Hugging Face Gradio application that generates text using a fine-tuned GPT-2 model. This project includes a custom implementation of the GPT architecture and a web interface for text generation.
## Technical Details
### Model Architecture
 Base Model: GPT-2 (124M parameters)
 Layers: 12 transformer blocks
 Attention Heads: 12
 Embedding Dimension: 768
 Maximum Sequence Length: 1024 tokens
 Vocabulary Size: 50,304 tokens
### Training Configuration
 Optimizer: AdamW with weight decay
 Learning Rate: Cosine schedule
 - Maximum LR: 6e-4
 - Minimum LR: 6e-5
 - Warmup Steps: 10
 Batch Size: 32
 Context Length: 128 tokens
 Training Steps: 10,000
 Early Stopping: Loss < 0.099999
 Gradient Clipping: 1.0
 Mixed Precision: bfloat16
### Features
 Checkpoint saving every 500 steps
 Training resumption from checkpoints
 Sample text generation during training
 Automatic device selection (CPU/CUDA/MPS)
 Flash Attention for efficient computation
### Web Interface
 Framework: Gradio
 Input Features:
 - Text prompt input
 - Sequence length slider (10-200)
 Output: Generated text display
 Example prompts included
## Installation
1. Clone the repository:

The model will be saved as `final_model.pt` and checkpoints will be saved in the `checkpoints` directory.

### Running the Web Interface

To start the Gradio interface:

The interface will be available at `http://localhost:7860`

## File Structure

## Dependencies
- torch
- gradio
- tiktoken

## Model Training Details

The model is trained on text data with the following features:
- Cosine learning rate schedule with warmup
- Weight decay for regularization
- Gradient clipping to prevent exploding gradients
- Checkpoint saving for training resumption
- Early stopping based on loss threshold
- Mixed precision training for efficiency

## Generation Parameters
- Temperature: 1.0 (default)
- Top-k sampling
- Support for special tokens
- Maximum generation length: configurable via interface

## Error Handling
- Robust token encoding/decoding
- Graceful handling of special tokens
- Fallback options for generation errors
- Informative error messages

## Performance Optimization
- Flash Attention implementation
- Mixed precision training
- Efficient batch processing
- Device-specific optimizations

## License
[Your License Here]

## Contributing
[Contribution Guidelines]

## Acknowledgments
- Based on the GPT-2 architecture
- Uses Hugging Face's tokenizer
- Implements techniques from the GPT-3 paper
## Training Log Summary (2025-02-15)

Training completed successfully with the following metrics:
- Initial loss: 10.9529
- Final loss: 0.0994
- Total steps: 5726
- Training duration: ~19 minutes
- Average tokens/sec: ~20,500

Key milestones:
- Loss < 7.0: Step 11 (6.9848)
- Loss < 6.0: Step 39 (5.9700)
- Loss < 5.0: Step 195 (4.9507)
- Loss < 4.0: Step N/A
- Loss < 3.0: Step N/A
- Loss < 2.0: Step N/A
- Loss < 1.0: Step 5725 (0.1009)
- Final loss: Step 5726 (0.0994)

Training was automatically stopped when loss threshold (0.1) was reached.
