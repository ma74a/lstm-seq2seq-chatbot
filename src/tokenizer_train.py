import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import List

from tokenizers import (Tokenizer, 
                        models, 
                        pre_tokenizers, 
                        decoders, 
                        trainers, 
                        processors)

from utils.config import Config

def get_training_data(data_path: str) -> List[str]:
    """Read the json data and put every conversation together

    Args:
        data_path (str): The json file path

    Returns:
        List[str]: List of each conversation
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    conversations = data["conversations"]
    
    corpus = []
    for conv in conversations:
        corpus.append(conv["input"])
        corpus.append(conv["output"])
        
    return corpus
        
def train_tokenizer(corpus: List[str]) -> Tokenizer:
    """Initialize a BPE tokenizer and train it and return it

    Args:
        corpus (List[str]): The List of conversation

    Returns:
        Tokenizer: The tokenizer we've trained on our data
    """
    # Initialize the tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Add pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # Add decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Initialize trainer
    trainer = trainers.BpeTrainer(
        vocab_size=Config.VOCAB_SIZE,
        special_tokens=Config.SPECIAL_TOKENS,
        min_frequency=2
    )
    
    tokenizer.train_from_iterator(iterator=corpus, trainer=trainer)
    
    # Add post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    
    return tokenizer
    
    
def main() -> None:
    """Main function
    """
    Config.create_directories()
    
    corpus = get_training_data(Config.DATA_PATH)
    
    tokenizer = train_tokenizer(corpus)
    tokenizer.save(str(Config.TOKENIZER_PATH))
    
    Config.VOCAB_SIZE = tokenizer.get_vocab_size()
    
if __name__ == "__main__":
    main()