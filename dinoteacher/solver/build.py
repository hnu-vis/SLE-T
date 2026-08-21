import torch
from functools import partial
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import StepParamScheduler, MultiStepParamScheduler




'''
ViT-S 12, ViT-B 12, ViT-L 24, ViT-G 40
'''

def get_dinov2_vit_layer_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    layer_id = int(name.split('.')[1]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)





def build_optimizer_with_layer_decay(model, cfg):








    
    
    num_vit_layers = cfg.MODEL.BACKBONE.NUM_LAYERS  
    lr_decay_rate = cfg.SOLVER.LR_DECAY_RATE  
    base_lr_dinov2 = cfg.SOLVER.DINOV2_BASE_LR  
    base_lr_adapter = cfg.SOLVER.ADAPTER_LR  
    base_lr_rpn = cfg.SOLVER.RPN_LR  
    base_lr_roi = cfg.SOLVER.ROI_HEAD_LR  
    weight_decay = cfg.SOLVER.WEIGHT_DECAY  
    
    
    lr_decay_func = partial(
        get_dinov2_vit_layer_lr_decay_rate,
        num_layers=num_vit_layers,
        lr_decay_rate=lr_decay_rate,
    )
    
    
    
    dinov2_layers = {f'blocks.{i}': [] for i in range(num_vit_layers)}
    dinov2_other = []  
    adapter_params = []
    rpn_params = []
    roi_head_params = []
    neck_params = []  
    
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        
        if ('backbone.encoder' in name) \
            and (name.split('.')[2] in \
                ['blocks', 'cls_token', 'mask_token', 'norm', 'patch_embed', 'pos_embed']):
                
            matched = False
            for i in range(num_vit_layers):
                if f'blocks.{i}' in name:
                    dinov2_layers[f'blocks.{i}'].append(param)
                    matched = True
                    break
            if not matched:
                
                dinov2_other.append(param)
                
        
        elif (name.split('.')[2] \
            in ['level_embed', 'spm', 'interactions', 'up', 'norm1', 'norm1', 'norm2', 'norm3', 'norm4']):
            
            adapter_params.append(param)
            
        
        elif 'rpn' in name.lower():
            rpn_params.append(param)
            
        
        elif 'roi_heads' in name.lower() or 'box_head' in name.lower() or 'mask_head' in name.lower():
            roi_head_params.append(param)
            
        
        elif 'fpn' in name.lower() or 'neck' in name.lower():
            neck_params.append(param)
            
        else:
            
            roi_head_params.append(param)
    
    
    param_groups = []
    
    
    for i in range(num_vit_layers):
        layer_name = f'blocks.{i}'
        if dinov2_layers[layer_name]:
            
            
            lr_factor = lr_decay_func(layer_name)
            
            param_groups.append({
                "params": dinov2_layers[layer_name],
                "lr": base_lr_dinov2 * lr_factor,
                "weight_decay": weight_decay,
                "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
            })
    
    
    if dinov2_other:
        param_groups.append({
            "params": dinov2_other,
            "lr": base_lr_dinov2,  
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    if adapter_params:
        param_groups.append({
            "params": adapter_params,
            "lr": base_lr_adapter,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    if neck_params:
        param_groups.append({
            "params": neck_params,
            "lr": cfg.SOLVER.NECK_LR if hasattr(cfg.SOLVER, 'NECK_LR') else base_lr_adapter,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    if rpn_params:
        param_groups.append({
            "params": rpn_params,
            "lr": base_lr_rpn,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    if roi_head_params:
        param_groups.append({
            "params": roi_head_params,
            "lr": base_lr_roi,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    print("=" * 60)
    print(" ( - DINOv2 ):")
    print(f"  DINOv2 layers: {num_vit_layers} ")
    for i in range(min(num_vit_layers, 3)):  
        layer_name = f'blocks.{i}'
        if dinov2_layers[layer_name]:
            lr = param_groups[i]['lr']
            print(f"    - {layer_name}: {len(dinov2_layers[layer_name])} , lr={lr:.2e}")
    if num_vit_layers > 6:
        print(f"    ... ()")
        for i in range(num_vit_layers-3, num_vit_layers):
            layer_name = f'blocks.{i}'
            if dinov2_layers[layer_name]:
                lr = param_groups[i]['lr']
                print(f"    - {layer_name}: {len(dinov2_layers[layer_name])} , lr={lr:.2e}")
    print(f"  DINOv2 other: {len(dinov2_other)} , lr={base_lr_dinov2:.2e}")
    print(f"  ViT Adapter: {len(adapter_params)} , lr={base_lr_adapter:.2e}")
    print(f"  Neck: {len(neck_params)} , lr={cfg.SOLVER.NECK_LR if hasattr(cfg.SOLVER, 'NECK_LR') else base_lr_adapter:.2e}")
    print(f"  RPN: {len(rpn_params)} , lr={base_lr_rpn:.2e}")
    print(f"  RoI Head: {len(roi_head_params)} , lr={base_lr_roi:.2e}")
    print("=" * 60)
    
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        eps=cfg.SOLVER.EPS if hasattr(cfg.SOLVER, 'EPS') else 1e-8,
    )
    
    return optimizer