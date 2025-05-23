import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

import json
from typing import Dict, List

from utils.config import Config

class ChatbotDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_path: str) -> None:
        """Read the data and encode the each conversation

        Args:
            data_path (str): The json data
            tokenizer_path (str): The trained tokenizer
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.conversations = data["conversations"]
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        self.inputs = []
        self.outputs = []
        for conv in self.conversations:
            input_encoding = self.tokenizer.encode(conv["input"])
            input_ids = self.tokenizer.encode(Config.SOS_TOKEN).ids+input_encoding.ids+self.tokenizer.encode(Config.EOS_TOKEN).ids
            
            output_encoding = self.tokenizer.encode(conv["output"])
            output_ids = self.tokenizer.encode(Config.SOS_TOKEN).ids+output_encoding.ids+self.tokenizer.encode(Config.EOS_TOKEN).ids
            
            input_ids = input_ids[:Config.MAX_INPUT_LENGTH]
            output_ids = output_ids[:Config.MAX_OUTPUT_LENGTH]
            
            self.inputs.append(torch.tensor(input_ids))
            self.outputs.append(torch.tensor(output_ids))
            
    def __len__(self) ->int:
        """Return the length of all conversation

        Returns:
            int: How many conversation
        """
        return len(self.conversations)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[idx],
            "input_len": len(self.inputs[idx]),
            "output_ids": self.outputs[idx],
            "output_len": len(self.outputs[idx])
        }
        
def create_data_loader(dataset: ChatbotDataset, batch_size: int=32, shuffle=False) -> DataLoader:
    """Create dataloader from ChatbotDataset class

    Args:
        dataset (ChatbotDataset): Object from ChatbotDataset class
        batch_size (int): The batch size of the data

    Returns:
        DataLoader: The dataloader we've created
    """
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset  # unwrap to get original ChatbotDataset
    else:
        base_dataset = dataset
    pad_id = base_dataset.tokenizer.encode(Config.PAD_TOKEN).ids[0]
    
    def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_id: int=pad_id) -> Dict[str, torch.Tensor]:
        """Prepares a batch of variable-length sequences for training by padding them to the same length.

            Args:
                batch (List[Dict[str, torch.Tensor]]): A list of dictionaries, each containing:
                    - 'input_ids': tensor of input token IDs
                    - 'input_len': original length of input sequence
                    - 'output_ids': tensor of output token IDs
                    - 'output_len': original length of output sequence

            Returns:
                Dict[str, torch.Tensor]: A dictionary containing:
                    - 'input_ids': padded input sequences [batch_size, max_input_len]
                    - 'output_ids': padded output sequences [batch_size, max_output_len]
                    - 'input_lengths': original lengths of input sequences
                    - 'output_lengths': original lengths of output sequences
            """
        # Get all sequences
        input_ids = [item['input_ids'] for item in batch]
        output_ids = [item['output_ids'] for item in batch]
        input_lengths = torch.tensor([item['input_len'] for item in batch])
        output_lengths = torch.tensor([item['output_len'] for item in batch])
        
        # Pad sequences
        # print(dataset)
        # print(dataset.tokenizer)
        # print(dataset.tokenizer.encode(Config.PAD_TOKEN).ids[0])
        # pad_id = dataset.tokenizer.encode(Config.PAD_TOKEN).ids[0]
        # pad_token_id = dataset.tokenizer.token_to_id(Config.PAD_TOKEN)
        # print(pad_token_id)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=pad_id
        )
        output_ids = torch.nn.utils.rnn.pad_sequence(
            output_ids, 
            batch_first=True, 
            padding_value=pad_id
        )
        
        return {
            'input_ids': input_ids,
            'output_ids': output_ids,
            'input_lengths': input_lengths,
            'output_lengths': output_lengths
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    ) 
        
        
if __name__ == "__main__":
    d = ChatbotDataset(data_path=Config.DATA_PATH,tokenizer_path=Config.TOKENIZER_PATH)
    loader = create_data_loader(dataset=d, batch_size=32)
    first_batch = next(iter(loader))
    # print(first_batch["input_ids"])
    # print(d.tokenizer)
#     first_seq = first_batch["input_ids"][0]
#     # print(first_seq)
#     first_seq = first_seq.tolist()
#     tokenizer = Tokenizer.from_file(str(Config.TOKENIZER_PATH))
#     decoded = tokenizer.decode(first_seq, skip_special_tokens=True)
#     print(decoded)