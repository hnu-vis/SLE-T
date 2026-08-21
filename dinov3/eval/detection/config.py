




from dataclasses import dataclass

from .models.position_encoding import PositionEncoding


@dataclass(kw_only=True)
class DetectionHeadConfig:
    num_classes: int = 91  
    
    with_box_refine: bool = True
    two_stage: bool = True
    
    mixed_selection: bool = True
    look_forward_twice: bool = True  
    
    k_one2many: int = 6  
    lambda_one2many: float = 1.0
    num_queries_one2one: int = 300  
    num_queries_one2many: int = 1500  
    """
    Absolute coordinates & box regression reparameterization.
    If true, we use absolute coordindates & reparameterization for bounding boxes.
    """
    reparam: bool = True
    topk: int = 100

    
    
    position_embedding: PositionEncoding = PositionEncoding.SINE
    num_feature_levels: int = 1  

    
    dec_layers: int = 6  
    dim_feedforward: int = 2048  
    hidden_dim: int = 256  
    dropout: float = 0.0  
    nheads: int = 8  
    norm_type: str = "pre_norm"

    
    aux_loss: bool = True  

    
    proposal_feature_levels: int = 4  
    proposal_min_size: int = 50
    
    decoder_type: str = "global_rpe_decomp"  
    decoder_use_checkpoint: bool = False
    decoder_rpe_hidden_dim: int = 512
    decoder_rpe_type: str = "linear"

    
    add_transformer_encoder: bool = True
    num_encoder_layers: int = 6
    layers_to_use: list[int] | None = None
    blocks_to_train: list[int] | None = None
    n_windows_sqrt: int = 0
    proposal_in_stride: int | None = None
    proposal_tgt_strides: list[int] | None = None
    backbone_use_layernorm: bool = False  
