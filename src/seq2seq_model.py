import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple

from utils.config import Config

# class Encoder(nn.Module):
#     def __init__(self, vocab_size: int,
#                         embedd_size: int,
#                         hidden_size: int,
#                         num_layer: int,
#                         dropout: float) -> None:
#         """This class takes tokenized input sequences, embeds them, and passes them through
#            an LSTM to produce hidden and cell states that represent the input sequence.

#         Args:
#             vocab_size (int): Number of tokens in the vocabulary.
#             embedd_size (int): Dimensionality of the embedding vectors.
#             hidden_size (int): Number of features in the hidden state of the LSTM.
#             num_layer (int): Number of LSTM layer
#             dropout (float): Dropout value
#         """
#         super().__init__()
#         self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
#                                             embedding_dim=embedd_size)
#         self.lstm = nn.LSTM(input_size=embedd_size,
#                             hidden_size=hidden_size,
#                             num_layers=num_layer,
#                             batch_first=True,
#                             dropout=dropout if num_layer > 1 else 0)
#         self.dropout = nn.Dropout(p=dropout)
        
#     # input_seq: [batch_size, input_seq]
#     def forward(self, input_ids: Tensor, 
#                       input_len: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
#         """Forward pass of the encoder with packed padded sequences.

#         Args:
#             input_ids (Tensor): Input tokens from tokenizer
#             input_len (Tensor): Each input len

#         Returns:
#             Tuple[Tensor, Tuple[Tensor, Tensor]]: Outputs, hidden_state, cell_state
#         """
#         embedding = self.dropout(self.embedding_layer(input_ids))
        
#         packed = nn.utils.rnn.pack_padded_sequence(
#             input=embedding,
#             lengths=input_len.cpu(),
#             batch_first=True,
#             enforce_sorted=False
#         )
#         outputs, (hidden_state, cell_state) = self.lstm(packed)
        
#         outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
#         # outputs: [batch_size, max_seq_len, hidden_size]
#         # hidden, cell [num_layers, batch_size, hidden_size]
#         return outputs, (hidden_state, cell_state)

class Encoder(nn.Module):
    def __init__(self, vocab_size: int,
                       embedd_size: int,
                       hidden_size: int,
                       num_layer: int,
                       dropout: float,
                       attention_heads: int = 4) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedd_size)
        self.lstm = nn.LSTM(
            input_size=embedd_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
            dropout=dropout if num_layer > 1 else 0
        )
        self.dropout = nn.Dropout(p=dropout)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attention_heads, batch_first=True)

    def forward(self, input_ids: Tensor, input_len: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        embedding = self.dropout(self.embedding_layer(input_ids))
        
        packed = nn.utils.rnn.pack_padded_sequence(embedding, input_len.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden_state, cell_state) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Apply self-attention
        attention_output, _ = self.attention(outputs, outputs, outputs)  # Q = K = V = LSTM outputs
        
        return attention_output, (hidden_state, cell_state)

    
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
    def forward(self, input_ids: Tensor, hidden_state: Tensor,
                      cell_state: Tensor) -> Tuple[float, Tuple[Tensor, Tensor]]:
        """Forward pass for decoder

        Args:
            input_ids (Tensor): The input of length one
            hidden_state (Tensor): Hidden state of the encoder
            cell_state (Tensor): Cell state of the encoder

        Returns:
            Tuple[float, Tuple[Tensor, Tensor]]: The prediction, Hidden state, Cell state
        """
        
        # [batch_size, 1, embed_size]
        embedding = self.dropout(self.embedding(input_ids))
        
        # outputs: [batch_size, 1, hidden_state]
        # hidden, cell: [num_layer, batch_size, hidden_state]
        outputs, (hidden_state, cell_state) = self.lstm(embedding, (hidden_state, cell_state))
        
        # [batch_size, 1, vocab_size]
        predictions = self.fc(outputs)
        
        return predictions, (hidden_state, cell_state)
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    # input_ids: [batch_size, seq_len]
    # output_ids; [batch_size, seq_len]
    def forward(self, input_ids: Tensor, input_len: Tensor,
                output_ids: Tensor, teacher_forcing_ratio: float) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = input_ids.size(0)
        output_len = output_ids.size(1)
        vocab_size = Config.VOCAB_SIZE
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, output_len, vocab_size).to(Config.DEVICE)
        
        encoder_outputs, (hidded, cell) = self.encoder(input_ids, input_len)
        
        # First decoder input is SOS
        # decoder_input = output_ids[:, 0].unsqueeze(0)
        decoder_input = output_ids[:, 0:1]
        
        # print(decoder_input.shape)
        for t in range(1, output_len):
            predictions, (hidden, cell) = self.decoder(decoder_input, hidded, cell)
            outputs[:, t:t+1, :] = predictions
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = predictions.argmax(2)
            decoder_input = output_ids[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs
    

    