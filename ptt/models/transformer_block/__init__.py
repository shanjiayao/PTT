# -*- coding: utf-8 -*-
from .multitransformer import MulTransformerBlock
from .variants import TransformerBlock, TransformerBlockALL, TransformerBlockBackbone, \
    TransformerBlockSTD, TransformerBlockCosine, TransformerBlockMLP, TransformerBlockOffset, \
    CrossAttentionBlock

__all__ = {
    'MulTransformerBlock': MulTransformerBlock,
    'TransformerBlock': TransformerBlock,
    'TransformerBlockALL': TransformerBlockALL,
    'TransformerBlockBackbone': TransformerBlockBackbone,
    'TransformerBlockCosine': TransformerBlockCosine,
    'TransformerBlockMLP': TransformerBlockMLP,
    'TransformerBlockOffset': TransformerBlockOffset,
    'TransformerBlockSTD': TransformerBlockSTD,
    'CrossAttentionBlock': CrossAttentionBlock
}


def build_transformer(model_cfg):
    return __all__[model_cfg.NAME](
        d_points=model_cfg.DIM_INPUT,
        d_model=model_cfg.DIM_MODEL,
        k=model_cfg.KNN,
        heads=model_cfg.N_HEADS,
        layers=model_cfg.N_LAYERS,
    )
