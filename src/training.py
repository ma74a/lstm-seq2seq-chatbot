import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from utils.config import Config
from seq2seq_model import Encoder, Decoder, Seq2Seq
from dataset import ChatbotDataset, create_data_loader
# from utils.visualize import plot_loss


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc="training"):
        input_ids = batch["input_ids"].to(device)
        input_len = batch["input_lengths"].to(device)
        target_ids = batch["output_ids"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, input_len, target_ids, Config.TEACHER_FORCING_RATIO)
        
        # Calculate loss (ignore padding tokens)
        # outputs shape: [batch_size, seq_len, vocab_size]
        # target shape: [batch_size, seq_len]
        outputs = outputs[:, 1:].reshape(-1, outputs.shape[-1])
        target = target_ids[:, 1:].reshape(-1)
        loss = criterion(outputs, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(train_loader)
                

def validate(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"validation"):
            input_ids = batch["input_ids"].to(device)
            input_len = batch["input_lengths"].to(device)
            target_ids = batch["output_ids"].to(device)
            
            outputs = model(input_ids, input_len, target_ids, 0)  # No teacher forcing during validation
            
            outputs = outputs[:, 1:].reshape(-1, outputs.shape[-1])
            target = target_ids[:, 1:].reshape(-1)
            loss = criterion(outputs, target)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(val_loader)
     

def main():
    Config.create_directories()
    
    # Create a ChatbotDataset object
    dataset = ChatbotDataset(Config.DATA_PATH, Config.TOKENIZER_PATH)
    
    # Split the data into train and val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(train_size, val_size)
    
    train_loader = create_data_loader(dataset=train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = create_data_loader(dataset=val_dataset)
    
    # Create the model
    encoder = Encoder(
        vocab_size=Config.VOCAB_SIZE,
        embedd_size=Config.EMBED_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layer=Config.NUM_LAYER,
        dropout=Config.DROPOUT
    )
    decoder = Decoder(
        vocab_size=Config.VOCAB_SIZE,
        embedd_size=Config.EMBED_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layer=Config.NUM_LAYER,
        dropout=Config.DROPOUT
    )
    model = Seq2Seq(encoder, decoder)
    model = model.to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    pad_token = dataset.tokenizer.encode(Config.PAD_TOKEN).ids[0]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader,optimizer, criterion, Config.DEVICE)
        val_loss = validate(model, val_loader, criterion, Config.DEVICE)
        
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': val_loss,
            }, str(Config.CHECKPOINT_DIR / 'best_model.pt'))
        
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': val_loss,
            }, str(Config.CHECKPOINT_DIR / f'model_epoch_{epoch+1}.pt'))
    
            print(f'epoch: {epoch}\tTrain Loss: {train_loss:.3f} | Valid Loss: {val_loss:.3f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            
    # plot_loss(train_losses, val_losses)
            
if __name__ == "__main__":
    main()