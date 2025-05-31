import torch
from pathlib import Path

class Config:
    # Json data path
    DATA_PATH = "/home/mahmoud/final_project/lstm-seq2seq-chatbot/data/furniture_chatbot_dataset.json"
    
    # LSTM Model Parameters
    EMBED_SIZE = 512
    HIDDEN_SIZE = 512
    NUM_LAYER = 1
    DROPOUT = 0.7
    
    # Hyper Parameters
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.0001
    TEACHER_FORCING_RATIO = 0.7
    
    # Tokenizer Parameters
    MAX_INPUT_LENGTH = 128
    MAX_OUTPUT_LENGTH = 256
    VOCAB_SIZE = 6000
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    SOS_TOKEN = "[SOS]"
    EOS_TOKEN = "[EOS]"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    TOKENIZER_PATH = BASE_DIR / "tokenized_data" / "furniture_tokenizer.json"
    
    @classmethod
    def create_directories(cls):
        cls.CHECKPOINT_DIR.mkdir(exist_ok=True)
        cls.TOKENIZER_PATH.parent.mkdir(exist_ok=True) 