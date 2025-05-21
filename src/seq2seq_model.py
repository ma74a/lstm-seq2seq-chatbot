import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from typing import Tuple

class Encoder(nn.Module):
    def __init__(self, vocab_size: int,
                        embedd_size: int,
                        hidden_size: int,
                        num_layer: int,
                        dropout: float) -> None:
        """This class takes tokenized input sequences, embeds them, and passes them through
           an LSTM to produce hidden and cell states that represent the input sequence.

        Args:
            vocab_size (int): Number of tokens in the vocabulary.
            embedd_size (int): Dimensionality of the embedding vectors.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            num_layer (int): Number of LSTM layer
            dropout (float): Dropout value
        """
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embedd_size)
        self.lstm = nn.LSTM(input_size=embedd_size,
                            hidden_size=hidden_size,
                            num_layers=num_layer,
                            batch_first=True,
                            dropout=dropout if num_layer > 1 else 0)
        self.dropout = nn.Dropout(p=dropout)
        
    # input_seq: [batch_size, input_seq]
    def forward(self, input_ids: torch.Tensor, 
                      input_len: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the encoder with packed padded sequences.

        Args:
            input_ids (torch.Tensor): Input tokens from tokenizer
            input_len (torch.Tensor): Each input len

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Outputs, hidden_state, cell_state
        """
        embedding = self.dropout(self.embedding_layer(input_ids))
        
        packed = nn.utils.rnn.pack_padded_sequence(
            input=embedding,
            lengths=input_len.cpu(),
            batch_first=True,
            enforce_sorted=True
        )
        outputs, (hidden_state, cell_state) = self.lstm(packed)
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # outputs: [batch_size, max_seq_len, hidden_size]
        # hidden, cell [num_layers, batch_size, hidden_size]
        return outputs, (hidden_state, cell_state)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size: int,
                       embedd_size: int,
                       hidden_size: int,
                       num_layer: int,
                       dropout: float) -> None:
        """The Decoder class

        Args:
            vocab_size (int): Number of tokens in the vocabulary.
            embedd_size (int): Dimensionality of the embedding vectors.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            num_layer (int): Number of LSTM layer
            dropout (float): Dropout value
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedd_size)
        self.lstm = nn.LSTM(
                input_size=embedd_size,
                hidden_size=hidden_size,
                num_layers=num_layer,
                batch_first=True,
                dropout=dropout if num_layer > 1 else 0
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        
    # input_ids: [batch_size, 1]
    def forward(self, input_ids: torch.Tensor, hidden_state: torch.Tensor,
                      cell_state: torch.Tensor) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor]]:
        
        # [batch_size, 1, embed_size]
        embedding = self.dropout(self.embedding(input_ids))
        
        # outputs: [batch_size, 1, hidden_state]
        # hidden, cell: [num_layer, batch_size, hidden_state]
        outputs, (hidden_state, cell_state) = self.lstm(embedding, (hidden_state, cell_state))
        
        # [batch_size, 1, vocab_size]
        predictions = self.fc(outputs)
        
        return predictions, (hidden_state, cell_state)