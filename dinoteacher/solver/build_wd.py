import torch
from functools import partial
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.solver import WarmupParamScheduler


from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import (
    StepParamScheduler, 
    MultiStepParamScheduler, 
    CosineParamScheduler,
    LinearParamScheduler
)



'''
ViT-S 12, ViT-B 12, ViT-L 24, ViT-G 40
'''

def get_dinov2_vit_layer_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    layer_id = int(name.split('.')[1]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def build_optimizer_with_layer_decay_dino(model, cfg):











    
    
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
    
    
    
    embedding_params = {'weights': [], 'biases': []}
    
    
    dinov2_layers = {f'blocks.{i}': {'weights': [], 'biases': []} for i in range(num_vit_layers)}
    
    
    dinov2_other = {'weights': [], 'biases': []}
    
    
    adapter_params = {'weights': [], 'biases': []}
    rpn_params = {'weights': [], 'biases': []}
    roi_head_params = {'weights': [], 'biases': []}
    neck_params = {'weights': [], 'biases': []}
    
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        
        is_bias = 'bias' in name.lower()
        
        
        if ('backbone.encoder' in name) \
            and (name.split('.')[2] in \
                ['blocks', 'cls_token', 'mask_token', 'norm', 'patch_embed', 'pos_embed']):
                
            
            if any(x in name for x in ['mask_token', 'cls_token', 'pos_embed', 'patch_embed']):
                if is_bias:
                    embedding_params['biases'].append(param)
                else:
                    embedding_params['weights'].append(param)
            
            
            elif 'blocks' in name:
                matched = False
                for i in range(num_vit_layers):
                    if f'blocks.{i}' in name:
                        if is_bias:
                            dinov2_layers[f'blocks.{i}']['biases'].append(param)
                        else:
                            dinov2_layers[f'blocks.{i}']['weights'].append(param)
                        matched = True
                        break
                if not matched:
                    
                    if is_bias:
                        dinov2_other['biases'].append(param)
                    else:
                        dinov2_other['weights'].append(param)
            
            
            else:
                if is_bias:
                    dinov2_other['biases'].append(param)
                else:
                    dinov2_other['weights'].append(param)
                
        
        elif (name.split('.')[2] \
            in ['level_embed', 'spm', 'interactions', 'up', 'norm1', 'norm1', 'norm2', 'norm3', 'norm4']):
            
            if is_bias:
                adapter_params['biases'].append(param)
            else:
                adapter_params['weights'].append(param)
            
        
        elif 'rpn' in name.lower():
            if is_bias:
                rpn_params['biases'].append(param)
            else:
                rpn_params['weights'].append(param)
            
        
        elif 'roi_heads' in name.lower() or 'box_head' in name.lower() or 'mask_head' in name.lower():
            if is_bias:
                roi_head_params['biases'].append(param)
            else:
                roi_head_params['weights'].append(param)
            
        
        elif 'fpn' in name.lower() or 'neck' in name.lower():
            if is_bias:
                neck_params['biases'].append(param)
            else:
                neck_params['weights'].append(param)
            
        else:
            
            if is_bias:
                roi_head_params['biases'].append(param)
            else:
                roi_head_params['weights'].append(param)
    
    
    param_groups = []
    
    
    
    def get_layer_lr(layer_index, base_lr):




        if layer_index == -1:
            
            
            
            embedding_lr_factor = lr_decay_rate ** num_vit_layers
            return base_lr * embedding_lr_factor
        else:
            
            
            dummy_name = f'blocks.{layer_index}'
            lr_factor = lr_decay_func(dummy_name)
            return base_lr * lr_factor
    
    
    if embedding_params['weights']:
        embedding_lr = get_layer_lr(-1, base_lr_dinov2)
        param_groups.append({
            "params": embedding_params['weights'],
            "lr": embedding_lr,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if embedding_params['biases']:
        embedding_lr = get_layer_lr(-1, base_lr_dinov2)
        param_groups.append({
            "params": embedding_params['biases'],
            "lr": embedding_lr,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    for i in range(num_vit_layers):
        layer_name = f'blocks.{i}'
        layer_params = dinov2_layers[layer_name]
        
        if layer_params['weights'] or layer_params['biases']:
            layer_lr = get_layer_lr(i, base_lr_dinov2)
            
            
            if layer_params['weights']:
                param_groups.append({
                    "params": layer_params['weights'],
                    "lr": layer_lr,
                    "weight_decay": weight_decay,
                    "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
                })
            
            
            if layer_params['biases']:
                param_groups.append({
                    "params": layer_params['biases'],
                    "lr": layer_lr,
                    "weight_decay": 0.0,
                    "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
                })
    
    
    if dinov2_other['weights']:
        param_groups.append({
            "params": dinov2_other['weights'],
            "lr": base_lr_dinov2,  
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if dinov2_other['biases']:
        param_groups.append({
            "params": dinov2_other['biases'],
            "lr": base_lr_dinov2,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    if adapter_params['weights']:
        param_groups.append({
            "params": adapter_params['weights'],
            "lr": base_lr_adapter,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if adapter_params['biases']:
        param_groups.append({
            "params": adapter_params['biases'],
            "lr": base_lr_adapter,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    neck_lr = cfg.SOLVER.NECK_LR if hasattr(cfg.SOLVER, 'NECK_LR') else base_lr_adapter
    if neck_params['weights']:
        param_groups.append({
            "params": neck_params['weights'],
            "lr": neck_lr,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if neck_params['biases']:
        param_groups.append({
            "params": neck_params['biases'],
            "lr": neck_lr,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    if rpn_params['weights']:
        param_groups.append({
            "params": rpn_params['weights'],
            "lr": base_lr_rpn,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if rpn_params['biases']:
        param_groups.append({
            "params": rpn_params['biases'],
            "lr": base_lr_rpn,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    if roi_head_params['weights']:
        param_groups.append({
            "params": roi_head_params['weights'],
            "lr": base_lr_roi,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if roi_head_params['biases']:
        param_groups.append({
            "params": roi_head_params['biases'],
            "lr": base_lr_roi,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    print("=" * 80)
    print(" ( - DINOv2  with Embedding Decay):")
    print(f"   (DINOv2): {base_lr_dinov2:.2e}")
    print(f"  : {lr_decay_rate}")
    print(f"  ViT : {num_vit_layers}")
    
    
    num_embed_weights = len(embedding_params['weights'])
    num_embed_biases = len(embedding_params['biases'])
    if num_embed_weights + num_embed_biases > 0:
        embedding_lr = get_layer_lr(-1, base_lr_dinov2)
        print(f"\n  [Embedding] mask_token, cls_token, pos_embed, patch_embed:")
        print(f"    - weights: {num_embed_weights} , lr={embedding_lr:.2e}")
        print(f"    - biases: {num_embed_biases} , lr={embedding_lr:.2e}")
    
    
    print(f"\n  [DINOv2 Blocks] ():")
    for i in range(min(num_vit_layers, 4)):  
        layer_params = dinov2_layers[f'blocks.{i}']
        num_weights = len(layer_params['weights'])
        num_biases = len(layer_params['biases'])
        if num_weights + num_biases > 0:
            layer_lr = get_layer_lr(i, base_lr_dinov2)
            print(f"    blocks.{i}: weights={num_weights}, biases={num_biases}, lr={layer_lr:.2e}")
    
    if num_vit_layers > 8:
        print("    ...")
        for i in range(num_vit_layers-4, num_vit_layers):  
            layer_params = dinov2_layers[f'blocks.{i}']
            num_weights = len(layer_params['weights'])
            num_biases = len(layer_params['biases'])
            if num_weights + num_biases > 0:
                layer_lr = get_layer_lr(i, base_lr_dinov2)
                print(f"    blocks.{i}: weights={num_weights}, biases={num_biases}, lr={layer_lr:.2e}")
    
    
    print(f"\n  [DINOv2 Other] (norm, head ):")
    print(f"    - weights: {len(dinov2_other['weights'])} , lr={base_lr_dinov2:.2e}")
    print(f"    - biases: {len(dinov2_other['biases'])} , lr={base_lr_dinov2:.2e}")
    
    print(f"\n  [ViT Adapter]:")
    print(f"    - weights: {len(adapter_params['weights'])} , lr={base_lr_adapter:.2e}")
    print(f"    - biases: {len(adapter_params['biases'])} , lr={base_lr_adapter:.2e}")
    
    print(f"\n  [Neck]:")
    print(f"    - weights: {len(neck_params['weights'])} , lr={neck_lr:.2e}")
    print(f"    - biases: {len(neck_params['biases'])} , lr={neck_lr:.2e}")
    
    print(f"\n  [RPN]:")
    print(f"    - weights: {len(rpn_params['weights'])} , lr={base_lr_rpn:.2e}")
    print(f"    - biases: {len(rpn_params['biases'])} , lr={base_lr_rpn:.2e}")
    
    print(f"\n  [RoI Head]:")
    print(f"    - weights: {len(roi_head_params['weights'])} , lr={base_lr_roi:.2e}")
    print(f"    - biases: {len(roi_head_params['biases'])} , lr={base_lr_roi:.2e}")
    
    print(f"\n  : {len(param_groups)}")
    print("=" * 80)
    
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        eps=cfg.SOLVER.EPS if hasattr(cfg.SOLVER, 'EPS') else 1e-8,
    )
    
    return optimizer

def build_lr_scheduler_v2(cfg, optimizer):




    
    max_iter = cfg.SOLVER.MAX_ITER
    scheduler_type = cfg.SOLVER.LR_SCHEDULER
    
    
    
    
    if scheduler_type == 'Step':
        
        step_size = cfg.SOLVER.STEP_SIZE
        gamma = cfg.SOLVER.GAMMA
        
        
        num_steps = max_iter // step_size
        values = [gamma ** i for i in range(num_steps + 1)]
        milestones = [step_size * (i + 1) for i in range(num_steps)]
        
        base_scheduler = MultiStepParamScheduler(
            values=values,
            milestones=milestones,
            num_updates=max_iter,
        )
        
    elif scheduler_type == 'MultiStep':
        
        milestones = cfg.SOLVER.MILESTONES
        gamma = cfg.SOLVER.GAMMA
        
        
        values = [1.0]
        for i, milestone in enumerate(milestones):
            values.append(values[-1] * gamma)
        
        base_scheduler = MultiStepParamScheduler(
            values=values,
            milestones=milestones,
            num_updates=max_iter,
        )
        
    elif scheduler_type == 'Cosine':
        
        min_lr_factor = cfg.SOLVER.MIN_LR / cfg.SOLVER.BASE_LR if hasattr(cfg.SOLVER, 'MIN_LR') else 0
        
        base_scheduler = CosineParamScheduler(
            start_value=1.0,
            end_value=min_lr_factor,
            num_updates=max_iter,
        )
        
    elif scheduler_type == 'Linear':
        
        min_lr_factor = cfg.SOLVER.MIN_LR / cfg.SOLVER.BASE_LR if hasattr(cfg.SOLVER, 'MIN_LR') else 0
        
        base_scheduler = LinearParamScheduler(
            start_value=1.0,
            end_value=min_lr_factor,
            num_updates=max_iter,
        )
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    
    
    
    warmup_iters = cfg.SOLVER.WARMUP_ITERS
    warmup_factor = cfg.SOLVER.WARMUP_FACTOR
    warmup_method = cfg.SOLVER.WARMUP_METHOD
    
    if warmup_iters > 0:
        lr_scheduler = WarmupParamScheduler(
            scheduler=base_scheduler,
            warmup_length=warmup_iters / max_iter,
            warmup_factor=warmup_factor,
            warmup_method=warmup_method,
        )
    else:
        
        
        def lr_scheduler(iteration):
            return base_scheduler(iteration)
    
    return lr_scheduler


def add_custom_config(cfg):

    
    cfg.MODEL.BACKBONE.NUM_LAYERS = 12  
    
    
    cfg.SOLVER.OPTIMIZER = "AdamW"
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.WEIGHT_DECAY = 0.1
    cfg.SOLVER.BETA1 = 0.9
    cfg.SOLVER.BETA2 = 0.999
    cfg.SOLVER.EPS = 1e-8
    
    
    cfg.SOLVER.DINOV2_BASE_LR = 5e-6      
    cfg.SOLVER.LR_DECAY_RATE = 0.7        
    cfg.SOLVER.ADAPTER_LR = 1.25e-5       
    cfg.SOLVER.NECK_LR = 1.25e-5          
    cfg.SOLVER.RPN_LR = 2.5e-5            
    cfg.SOLVER.ROI_HEAD_LR = 5e-5         
    
    
    
    
    cfg.SOLVER.USE_FVCORE_SCHEDULER = True  
    cfg.SOLVER.LR_SCHEDULER = "MultiStep"   
    cfg.SOLVER.STEP_SIZE = 30000
    cfg.SOLVER.MILESTONES = [60000, 80000]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.MIN_LR = 0
    
    
    
    
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 9000
    cfg.SOLVER.WARMUP_METHOD = "linear"   
    
    
    
    
    return cfg