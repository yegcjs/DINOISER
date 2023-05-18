import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class TransformerDecoder(nn.Module):
    def __init__(self, args, config):
        super().__init__()
       
        self.pos_embedding = nn.Embedding(args.max_target_positions, config["hidden_dim"])
        self.trm = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                config["hidden_dim"],
                config["num_heads"],
                dim_feedforward=config["ffn_dim"],
                dropout=args.dropout,
                activation=F.gelu,
                batch_first=True,
                norm_first=config["dec_prenorm"],
            ),
            config["dec_num_layers"],
        )
        self.prenorm = config["dec_prenorm"]
        self.norm = nn.LayerNorm(config["hidden_dim"])
    
    def forward(self, x, padding_mask, time_embed, encoder_out=None):
        emb_pos = self.pos_embedding.weight[:x.shape[1]].unsqueeze(0)
        x = x + emb_pos
        if not self.prenorm:
            x = self.norm(x)
        if encoder_out is None:
            encoder_out = {
                "feature": torch.zeros(x.size(0), 1, x.size(-1)).to(x),
                "padding_mask": torch.zeros(x.size(0), 1).to(padding_mask)
            }
        x = self.trm(
            x, encoder_out["feature"], tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=encoder_out["padding_mask"]
        )
        if self.prenorm:
            x = self.norm(x)
        return x

class TransformerEncoderModel(nn.Module):
    def __init__(self, args, config, src_dict) -> None:
        super().__init__()
        max_position = max(args.max_source_positions, args.max_target_positions)
        self.register_buffer("position_ids", torch.arange(max_position).expand((1, -1)))
        
        self.config = config
        # encoder
        self.encoder = self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config["hidden_dim"],
                config["num_heads"],
                dim_feedforward=config["ffn_dim"],
                dropout=args.dropout,
                activation=F.gelu,
                batch_first=True,
                norm_first=config["enc_prenorm"]
            ),
            config["enc_num_layers"],
            nn.LayerNorm(config["hidden_dim"])
        )
        self.enc_token_embed = nn.Embedding(len(src_dict), config["hidden_dim"], padding_idx=src_dict.pad())
        self.enc_pos_embed = nn.Embedding(args.max_source_positions, config["hidden_dim"])
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input_tokens, padding_mask=None):
        if padding_mask is None:
            padding_mask = (input_tokens==self.enc_token_embed.padding_idx)
        position_ids = self.position_ids[:, : input_tokens.shape[-1] ]
        position_embed = self.enc_pos_embed(position_ids)
        token_embed = self.enc_token_embed(input_tokens)
        embedding = token_embed + position_embed
        embedding = self.dropout(embedding)
        feature = self.encoder(embedding, src_key_padding_mask=padding_mask)
        return {
            "feature": feature,
            "padding_mask": padding_mask
        }
