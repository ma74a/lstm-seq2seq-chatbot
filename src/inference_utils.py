import torch
from tokenizers import Tokenizer

from src.seq2seq_model import Encoder, Decoder, Seq2Seq
from utils.config import Config

def load_model(model_path: str) -> Seq2Seq:
    encoder = Encoder(
        vocab_size=Config.VOCAB_SIZE,
        embedd_size=Config.EMBED_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layer=Config.NUM_LAYER,
        dropout=0
    )
    decoder = Decoder(
        vocab_size=Config.VOCAB_SIZE,
        embedd_size=Config.EMBED_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layer=Config.NUM_LAYER,
        dropout=0
    )
    model = Seq2Seq(encoder=encoder, decoder=decoder)
    checkpoints = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoints["model_state_dict"])
    model.to(Config.DEVICE)
    return model


def generate_response(model: Seq2Seq, tokenizer: Tokenizer, user_input: str) -> str:
    # model.eval()
    input_encoding = tokenizer.encode(user_input)
    # print(input_encoding.ids)
    input_ids = tokenizer.encode(Config.SOS_TOKEN).ids+input_encoding.ids+tokenizer.encode(Config.EOS_TOKEN).ids
    # print(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(Config.DEVICE) # [1, seq]
    input_len = torch.tensor([len(input_ids[0])]).to(Config.DEVICE)
    
    with torch.no_grad():
        encoder_output, (hidden, cell) = model.encoder(input_ids, input_len)
        
        # start with sos for decoder
        decoder_input = torch.tensor([tokenizer.encode(Config.SOS_TOKEN).ids]) # tensor([[2]])
        # print(decoder_input)
        output_ids = []
        for _ in range(50):
            prediction, (hidden, cell) = model.decoder(decoder_input, hidden, cell)
            predicted_id = prediction.squeeze(0).argmax().item()
            if predicted_id == tokenizer.encode(Config.EOS_TOKEN).ids[0]:
                break
            output_ids.append(predicted_id)
            decoder_input = torch.tensor([[predicted_id]], device=Config.DEVICE)

    # Decode the output token ids to text
    return tokenizer.decode(output_ids)