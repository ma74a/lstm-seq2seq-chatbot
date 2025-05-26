# LSTM Seq2Seq Chatbot

A conversational AI chatbot built using LSTM (Long Short-Term Memory) sequence-to-sequence architecture, specifically designed for furniture-related conversations. The project includes training pipeline, inference capabilities, and a FastAPI web service for deployment.

## 🏗️ Architecture

The chatbot uses an encoder-decoder architecture with LSTM networks:

- **Encoder**: Processes input sequences and generates context representations
- **Decoder**: Generates responses token by token using attention mechanism
- **Tokenizer**: Custom BPE (Byte Pair Encoding) tokenizer trained on the dataset

## 📁 Project Structure

```
lstm-seq2seq-chatbot/
├── app/
│   └── main.py              # FastAPI web service
├── src/
│   ├── dataset.py           # Dataset loading and preprocessing
│   ├── inference_utils.py   # Model inference utilities
│   ├── seq2seq_model.py     # LSTM Encoder-Decoder model
│   ├── tokenizer_train.py   # Tokenizer training script
│   └── training.py          # Model training script
├── utils/
│   ├── config.py            # Configuration settings
│   └── visualize.py         # Training visualization
├── inference.py             # Command-line inference script
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/lstm-seq2seq-chatbot.git
cd lstm-seq2seq-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FastAPI dependencies (for web service):
```bash
pip install fastapi uvicorn
```

### Data Setup

1. Prepare your dataset in JSON format:
```json
{
  "conversations": [
    {
      "input": "What furniture do you have?",
      "output": "We have sofas, chairs, tables, and wardrobes available."
    },
    {
      "input": "How much is the sofa?",
      "output": "Our sofas range from $200 to $800 depending on the style and condition."
    }
  ]
}
```

2. Update the data path in `utils/config.py`:
```python
DATA_PATH = "path/to/your/furniture_chatbot_dataset.json"
```

## 🔧 Configuration

Key configuration parameters in `utils/config.py`:

```python
# Model Architecture
EMBED_SIZE = 512          # Embedding dimension
HIDDEN_SIZE = 512         # LSTM hidden size
NUM_LAYER = 2            # Number of LSTM layers
DROPOUT = 0.7            # Dropout rate

# Training Parameters
BATCH_SIZE = 64          # Training batch size
EPOCHS = 200             # Number of training epochs
LEARNING_RATE = 0.0001   # Learning rate
TEACHER_FORCING_RATIO = 0.7  # Teacher forcing probability

# Tokenizer Settings
VOCAB_SIZE = 5000        # Vocabulary size
MAX_INPUT_LENGTH = 128   # Maximum input sequence length
MAX_OUTPUT_LENGTH = 256  # Maximum output sequence length
```

## 🎯 Training

### Step 1: Train Tokenizer

```bash
python src/tokenizer_train.py
```

This creates a custom BPE tokenizer trained on your conversation data.

### Step 2: Train Model

```bash
python src/training.py
```

The training script will:
- Split data into train/validation sets (80/20)
- Train the seq2seq model
- Save checkpoints every 10 epochs
- Save the best model based on validation loss
- Generate training/validation loss plots

### Step 3: Monitor Training

Training progress is visualized with loss plots saved as `loss_plot.png`.

## 💬 Inference

### Command Line Interface

```bash
python inference.py
```

Interactive chat session:
```
Chatbot for Used furniture app
**********************
User: What sofas do you have?
Bot: We have leather sofas, fabric sofas, and sectional sofas available.
----------------------
User: quit
```

### Programmatic Usage

```python
from tokenizers import Tokenizer
from src.inference_utils import load_model, generate_response
from utils.config import Config

# Load model and tokenizer
tokenizer = Tokenizer.from_file(str(Config.TOKENIZER_PATH))
model = load_model(Config.CHECKPOINT_DIR / "best_model.pt")

# Generate response
response = generate_response(model, tokenizer, "What chairs do you have?")
print(response)
```

## 🌐 Web API

### Start FastAPI Server

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Health Check
```bash
GET http://localhost:8000/
```

#### Chat Endpoint
```bash
POST http://localhost:8000/chat
Content-Type: application/json

{
  "message": "What furniture do you have in stock?"
}
```

Response:
```json
{
  "response": "We have sofas, chairs, dining tables, and bedroom sets available."
}
```

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "How much is the dining table?"}'
```

## 📊 Model Performance

The model uses several techniques to improve performance:

- **Teacher Forcing**: During training, uses actual target tokens as decoder input
- **Gradient Clipping**: Prevents exploding gradients (max norm = 1.0)
- **Dropout Regularization**: Reduces overfitting
- **Packed Sequences**: Efficient handling of variable-length sequences
- **Cross-Entropy Loss**: Ignores padding tokens during loss calculation

## 🔧 Customization

### Adding New Special Tokens

Modify `SPECIAL_TOKENS` in `config.py`:
```python
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[CUSTOM]"]
```

### Adjusting Model Architecture

```python
# Increase model capacity
EMBED_SIZE = 768
HIDDEN_SIZE = 768
NUM_LAYER = 3

# Adjust sequence lengths
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 512
```

### Custom Dataset Format

Modify `ChatbotDataset` class in `src/dataset.py` to handle different data formats.

## 📈 Monitoring and Evaluation

### Training Metrics
- Training loss per epoch
- Validation loss per epoch
- Best model selection based on validation performance


## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `BATCH_SIZE` in config
2. **Poor response quality**: Increase training epochs or model size
3. **Slow inference**: Ensure model is on correct device (GPU/CPU)
4. **Tokenizer errors**: Verify tokenizer file path and format

### Debug Mode

Enable detailed logging by modifying the inference functions to print intermediate steps.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation

## 🚦 Future Enhancements

- [ ] Attention mechanism implementation
- [ ] Beam search decoding
- [ ] Docker containerization
- [ ] Model quantization for mobile deployment
