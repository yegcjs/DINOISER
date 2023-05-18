import json

from fairseq.models import register_model_architecture
from .diffusion import DiffusionCLMModel

import copy

base_transformer = {
    "hidden_dim": 512,
    "ffn_dim": 2048,
    "time_conditional_layernorm": False,
    "num_heads": 8,
    "enc_num_layers": 6,
    "enc_prenorm": False,
    "dec_num_layers": 6,
    "dec_prenorm": True,
    "dec_causal": False
}

base_vae = {
    "hidden_dim": 64,
    "embedding_init_std": 1e-3,
    "normalized_embedding": True,

    "inf_num_layers": 0,
    "gen_num_layers": 0,
    "std": 0
}

def wmt_base_model(args, vae_config, transformer_config):
    args.scheduler = getattr(args, "scheduler", "quadratic")
    args.dropout = getattr(args, "dropout", 0.3)
    args.latent_dim = getattr(args, "latent_dim", 64)
    
    args.vae_type = getattr(args, "vae_type", "vae")
    vae_config = copy.deepcopy(vae_config)
    vae_config["hidden_dim"] = args.latent_dim
    args.vae_config = json.dumps(vae_config)
    args.half_time_embedding_dim = getattr(args, "half_time_embedding-dim", 128)

    args.diffusion_model_type = getattr(args, "diffusion_model_type", "transformer")
    args.diffusion_model_config = json.dumps(transformer_config)

    args.model_output_type = getattr(args, "model_output_type", "noise")
    args.partial_diffusion = getattr(args, "partial_diffusion", False)
    args.self_conditioning = getattr(args, "self_conditioning", False)

    args.length_predict_type = getattr(args, "length_predict_type", "absolute")
    args.length_model_type = getattr(args, "length_model_type", "classification")

    args.tpu = False

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm")
def wmt_base_prenorm(args):
    wmt_base_model(args, base_vae, base_transformer)

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm_unnormalized")
def wmt_base_prenorm_unnormalized(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["normalized_embedding"] = False
    wmt_base_model(args, vae_config, base_transformer)

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm_init0_01")
def wmt_base_prenorm_init0_1(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 0.01
    wmt_base_model(args, vae_config, base_transformer)

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm_init0_01_unnormalized")
def wmt_base_prenorm_init0_01_unnormalized(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 0.01
    vae_config["normalized_embedding"] = False
    wmt_base_model(args, vae_config, base_transformer)

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm_init0_1")
def wmt_base_prenorm_init0_1(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 0.1
    wmt_base_model(args, vae_config, base_transformer)

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm_init0_1_unnormalized")
def wmt_base_prenorm_init0_1_unnormalized(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 0.1
    vae_config["normalized_embedding"] = False
    wmt_base_model(args, vae_config, base_transformer)

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm_init1")
def wmt_base_prenorm_init1(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 1
    wmt_base_model(args, vae_config, base_transformer)

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm_init1_unnormalized")
def wmt_base_prenorm_init1_unnormalized(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 1
    vae_config["normalized_embedding"] = False
    wmt_base_model(args, vae_config, base_transformer)

@register_model_architecture("diffusion_clm_model", "iwslt_base_prenorm")
def iwslt_base_prenorm(args):
    vae_config = copy.deepcopy(base_vae)
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "iwslt_base_prenorm_condln")
def iwslt_base_prenorm_condln(args):
    vae_config = copy.deepcopy(base_vae)
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    transformer_config["time_conditional_layernorm"] = True
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "iwslt_base_prenorm_init1_unnormalized")
def iwslt_base_prenorm_init1_unnormalized(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 1
    vae_config["normalized_embedding"] = False
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "iwslt_base_prenorm_init1_unnormalized_condln")
def iwslt_base_prenorm_init1_unnormalized_condln(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 1
    vae_config["normalized_embedding"] = False
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    transformer_config["time_conditional_layernorm"] = True
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm_init1_unnormalized_condln")
def wmt_base_prenorm_init1_unnormalized(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 1
    vae_config["normalized_embedding"] = False
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["time_conditional_layernorm"] = True
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm_condln")
def wmt_base_prenorm_condln(args):
    vae_config = copy.deepcopy(base_vae)
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["time_conditional_layernorm"] = True
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "wmt_base_postnorm_init1_unnormalized")
def wmt_base_postnorm_init1_unnormalized(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 1
    vae_config["normalized_embedding"] = False
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_prenorm"] = False
    wmt_base_model(args, vae_config, transformer_config)


@register_model_architecture("diffusion_clm_model", "iwslt_base_postnorm")
def iwslt_base_postnorm(args):
    vae_config = copy.deepcopy(base_vae)
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_prenorm"] = False
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "iwslt_base_postnorm_12_1")
def iwslt_base_postnorm_12_1(args):
    vae_config = copy.deepcopy(base_vae)
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_prenorm"] = False
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    transformer_config["enc_num_layers"] = 12
    transformer_config["dec_num_layers"] = 1
    wmt_base_model(args, vae_config, transformer_config)


@register_model_architecture("diffusion_clm_model", "iwslt_base_postnorm_8_4")
def iwslt_base_postnorm_8_4(args):
    vae_config = copy.deepcopy(base_vae)
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_prenorm"] = False
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    transformer_config["enc_num_layers"] = 8
    transformer_config["dec_num_layers"] = 4
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "iwslt_base_postnorm_10_2")
def iwslt_base_postnorm_10_2(args):
    vae_config = copy.deepcopy(base_vae)
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_prenorm"] = False
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    transformer_config["enc_num_layers"] = 10
    transformer_config["dec_num_layers"] = 2
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "iwslt_base_postnorm_condln")
def iwslt_base_postnorm_condln(args):
    vae_config = copy.deepcopy(base_vae)
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_prenorm"] = False
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    transformer_config["time_conditional_layernorm"] = True
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "iwslt_base_postnorm_init1_unnormalized")
def iwslt_base_postnorm_init1_unnormalized(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 1
    vae_config["normalized_embedding"] = False
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_prenorm"] = False
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "iwslt_base_postnorm_init1_unnormalized_condln")
def iwslt_base_postnorm_init1_unnormalized_condln(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 1
    vae_config["normalized_embedding"] = False
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_prenorm"] = False
    transformer_config["ffn_dim"] = 1024
    transformer_config["num_heads"] = 4
    transformer_config["time_conditional_layernorm"] = True
    wmt_base_model(args, vae_config, transformer_config)

@register_model_architecture("diffusion_clm_model", "wmt_base_prenorm_trained_std")
def wmt_base_prenorm_trained_std(args):
    vae_config = copy.deepcopy(base_vae)
    vae_config["std"] = "trained"
    wmt_base_model(args, vae_config, base_transformer)

@register_model_architecture("diffusion_clm_model", "wmt_base_postnorm")
def wmt_base_postnorm(args):
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_prenorm"] = False
    wmt_base_model(args, base_vae, transformer_config)

@register_model_architecture("diffusion_clm_model", "wmt_base_causal_prenorm")
def wmt_base_causal_prenorm(args):
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_causal"] = True
    wmt_base_model(args, base_vae, transformer_config)


@register_model_architecture("diffusion_clm_model", "wmt_base_causal_prenorm_init1_unnormalized")
def wmt_base_causal_prenorm_init1_unnormalized(args):
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_causal"] = True
    vae_config = copy.deepcopy(base_vae)
    vae_config["embedding_init_std"] = 1
    vae_config["normalized_embedding"] = False
    wmt_base_model(args, vae_config, transformer_config)


@register_model_architecture("diffusion_clm_model", "wmt_base_causal_postnorm")
def wmt_base_causal_postnorm(args):
    transformer_config = copy.deepcopy(base_transformer)
    transformer_config["dec_prenorm"] = False
    transformer_config["dec_causal"] = True
    wmt_base_model(args, base_vae, transformer_config)

# emb-<dim>d-(pre|post)norm-<init_std>[-cln][-sc]-