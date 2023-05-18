import torch
import torch.nn as nn
import torch.nn.functional as F

import json

""" vae config
    hidden_dim: int
    embedding_init_std: float
    normalized_embedding: true | false

    num_heads:
    ffn_dim:
    inf_num_layers:
    gen_num_layers:

    (with inf_num_layers==0, gen_num_layers==0, hidden_dim==latent_dim, we get embedding only model)

    vae:
    std: "trained" | float

    vq-vae:
    num_code: int
"""

class VAEBase(nn.Module):
    def __init__(self, args, config, tgt_dict) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(len(tgt_dict), config["hidden_dim"])
        self.embedding.weight.data.normal_(mean=0, std=config["embedding_init_std"])
        self.normalized_embedding = config["normalized_embedding"]

        self.register_buffer("position_ids", torch.arange(args.max_target_positions).expand((1, -1)))
        self.inf_pos_emb = (
            nn.Embedding(args.max_target_positions, config["hidden_dim"])
            if config["inf_num_layers"] > 0 else None
        )
        self.inference_net = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    config["hidden_dim"], 
                    config["num_heads"], 
                    config["ffn_dim"],
                    batch_first=True,
                    activation=F.gelu,
                ), 
                num_layers=config["inf_num_layers"],
                norm=nn.LayerNorm(config["hidden_dim"])
            ) 
            if config["inf_num_layers"] > 0
            else None
        )
        self.down_sampler = (
            nn.Sequential(
                nn.Linear(config["hidden_dim"], config["hidden_dim"]),
                nn.GELU(),
                nn.Linear(config["hidden_dim"], args.latent_dim)
            )
            if config["hidden_dim"] != args.latent_dim
            else nn.Identity()
        )
        self.up_sampler = (
            nn.Sequential(
                nn.Linear(args.latent_dim, config["hidden_dim"]),
                nn.GELU(),
                nn.Linear(config["hidden_dim"], config["hidden_dim"])
            )
            if config["hidden_dim"] != args.latent_dim
            else nn.Identity()
        ) 
        self.gen_pos_emb = (
            nn.Embedding(args.max_target_positions, config["hidden_dim"])
            if config["gen_num_layers"] > 0 else None
        )
        self.generator_net = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    config["hidden_dim"], 
                    config["num_heads"], 
                    config["ffn_dim"],
                    activation=F.gelu,
                    batch_first=True
                ), 
                num_layers=config["gen_num_layers"],
                norm=nn.LayerNorm(config["hidden_dim"])
            ) 
            if config["gen_num_layers"] > 0
            else None 
        )

    def get_embedding_weight(self):
        if self.normalized_embedding:
            scale = self.embedding.weight.size(-1) ** 0.5
            return F.normalize(self.embedding.weight, p=2) * scale
        else:
            return self.embedding.weight

    def clamp(self, x, padding_mask):
        penultimate_layer = self.generator_penultimate_layer(x, padding_mask)
        norm_layer = (penultimate_layer ** 2).sum(-1)
        embedding = self.get_embedding_weight()
        norm_emb = (embedding**2).sum(-1)
        dist = norm_layer[:, :, None] + norm_emb[None, None, :] - 2 * (penultimate_layer @ embedding.T)
        nn_indices = (-dist).topk(k=1, dim=-1).indices.squeeze()
        return embedding[nn_indices]

    def intepolate(self, x, padding_mask):
        logits = self.generator(x, padding_mask=padding_mask)["logits"]
        prob = F.softmax(logits, dim=-1)
        intepolated_embedding = prob @ self.embedding.weight
        return self.inference_with_embedding(intepolated_embedding, padding_mask)

    def inference_with_embedding(self, x, padding_mask):
        if self.inf_pos_emb is not None:
            position_ids = self.position_ids[:, : padding_mask.shape[-1]]
            x += self.inf_pos_emb(position_ids)
        if self.inference_net is not None:
            x = self.inference_net(x, src_key_padding_mask=padding_mask)
        down_sampled_x = self.down_sampler(x)
        return down_sampled_x, x

    def inference(self, tokens, padding_mask):
        embedding = self.get_embedding_weight()
        x = embedding[tokens]
        return self.inference_with_embedding(x, padding_mask)    
    
    def generator_penultimate_layer(self, latent, padding_mask):
        x = self.up_sampler(latent)
        if self.gen_pos_emb is not None:
            position_ids = self.position_ids[:, : latent.shape[1]]
            x += self.inf_pos_emb(position_ids)
        if self.generator_net is not None:
            x = self.generator_net(x, src_key_padding_mask=padding_mask)
        return x

    def generator(self, latent, padding_mask):
        x = self.generator_penultimate_layer(latent, padding_mask)
        embedding = self.get_embedding_weight()
        logits = F.linear(x, embedding)

        return {
            "logits": logits,
            "prediction": logits.argmax(-1)
        }    

class VAE(VAEBase):
    def __init__(self, args, config, tgt_dict) -> None:
        super().__init__(args, config, tgt_dict)
        config = self.config
        if config["std"] == "trained":
            self.down_sampler_std = nn.Sequential(
                nn.Linear(config["hidden_dim"], config["hidden_dim"]),
                nn.GELU(),
                nn.Linear(config["hidden_dim"], args.latent_dim),
                nn.Softplus()
            )
        else:
            self.fixed_std = float(config["std"])
    
    def inference(self, tokens, padding_mask):
        mean, feature = super().inference(tokens, padding_mask)
        if hasattr(self, "down_sampler_std"):
            std = self.down_sampler_std(feature)
        else:
            std = torch.ones_like(mean) * self.fixed_std
        return {
            "latent": mean + std * torch.randn_like(mean),
            "mean": mean,
            "std": std
        }



class VQVAE(VAEBase):
    def __init__(self, args, config, tgt_dict) -> None:
        super().__init__(args, config, tgt_dict)
        self.num_code = self.config["num_code"]
        self.code = nn.Embedding(self.num_code, args.latent_dim)


    def clamp_indices(self, x_start):
        ori_shape = x_start.shape[:-1]
        emb_norm = (self.code.weight**2).sum(-1).view(-1, 1) #vocab
        text_emb_t = torch.transpose(x_start.view(-1, x_start.size(-1)), 0, 1) #d, bsz*seqlen
        arr_norm = (x_start ** 2).sum(-1).view(-1, 1) #bsz*seqlen, 1
        # print(emb_norm.shape, arr_norm.shape)
        dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.code.weight, text_emb_t) #(vocab, d) x (d, bsz*seqlen)
        dist = torch.clamp(dist, 0.0, 1e12)
        return torch.topk(-dist, k=1, dim=0).indices.view(ori_shape)

    def clamp(self, x_start):
        nn_indices = self.clamp_indices(x_start)
        return self.code(nn_indices) 

    def inference(self, tokens, padding_mask):
        mean, x = super().inference(tokens, padding_mask)
        latent = self.clamp(mean)
        return {
            "latent": mean + (latent.detach() - mean.detach()),
            "quantized": latent,
            "unquantized": mean
        }

    def generator(self, latent, padding_mask):
        latent = self.clamp(latent)
        return super().generator(latent, padding_mask)

