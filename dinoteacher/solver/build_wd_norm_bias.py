import torch
from functools import partial
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import (
    MultiStepParamScheduler, 
    CosineParamScheduler,
    LinearParamScheduler
)
from detectron2.solver.build import reduce_param_groups
from detectron2.solver import LRMultiplier


from IPython import embed


'''
ViT-S 12, ViT-B 12, ViT-L 24, ViT-G 40
'''

def get_dinov2_vit_layer_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    layer_id = int(name.split('.')[1]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def build_optimizer_with_layer_decay(model, cfg):













    
    
    
    num_layers_dict = {
        "dinov2_vitb14_light" : [12, ],
        "dinov3_vitb16_light" : [12, ],
        "dinov3_vitb16_lt_vgg_light" : [12, ],
        "dinov3_va_lt_b16_vgg_light" : [12, ],
        "dinov2_vitg14_light" : [40, ],
        "dinov2_vitl14_light" : [24, ],
    }
    layer_keys = '_'.join([cfg.SEMISUPNET.DINO_BBONE_MODEL, cfg.SEMISUPNET.VA_TYPE])
    num_vit_layers = num_layers_dict[layer_keys][0]
    
    
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
    
    
    def is_norm_param(name):

        norm_keywords = ['norm', 'ln', 'layer_norm', 'batch_norm', 'batchnorm']
        return any(keyword in name.lower() for keyword in norm_keywords)
    
    
    def is_bias_param(name):
        return 'bias' in name.lower()
    
    def is_basic_param(name):
        basic_keywords = ['level_embed', 'pos_embed', 'rope_embed', 'storage_tokens', 'storage_token']
        return any(keyword in name.lower() for keyword in basic_keywords)
    
    
    def skip_weight_decay(name):

        return is_bias_param(name) or is_norm_param(name) or is_basic_param(name)
    
    
    
    embedding_params = {'with_wd': [], 'without_wd': []}  
    
    
    dinov2_layers = {f'blocks.{i}': {'with_wd': [], 'without_wd': []} for i in range(num_vit_layers)}
    
    
    dinov2_other = {'with_wd': [], 'without_wd': []}
    
    
    adapter_params = {'with_wd': [], 'without_wd': []}
    rpn_params = {'with_wd': [], 'without_wd': []}
    roi_head_params = {'with_wd': [], 'without_wd': []}
    neck_params = {'with_wd': [], 'without_wd': []}
    
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        
        
        
        
        apply_wd = not skip_weight_decay(name)
        
        
        
        
        
        dino_param_roots = [
            'blocks', 'cls_token', 'mask_token', 'norm', 'patch_embed',
            'pos_embed', 'rope_embed', 'storage_tokens', 'storage_token'
        ]
        dino_embedding_keywords = [
            'mask_token', 'cls_token', 'pos_embed', 'patch_embed',
            'rope_embed', 'storage_tokens', 'storage_token'
        ]
        if ('backbone.encoder' in name) \
            and (name.split('.')[2] in \
                dino_param_roots):
                
            
            if any(x in name for x in dino_embedding_keywords):
                if apply_wd:
                    embedding_params['with_wd'].append(param)
                else:
                    embedding_params['without_wd'].append(param)
            
            
            elif 'blocks' in name:
                matched = False
                for i in range(num_vit_layers):
                    
                    
                    if f'blocks.{i}.' in name:
                        if apply_wd:
                            dinov2_layers[f'blocks.{i}']['with_wd'].append(param)
                        else:
                            dinov2_layers[f'blocks.{i}']['without_wd'].append(param)
                        matched = True
                        break
                if not matched:
                    
                    if apply_wd:
                        dinov2_other['with_wd'].append(param)
                    else:
                        dinov2_other['without_wd'].append(param)
            
            
            else:
                if apply_wd:
                    dinov2_other['with_wd'].append(param)
                else:
                    dinov2_other['without_wd'].append(param)
                
        
        elif (name.split('.')[2] \
            
            in ['level_embed', 'spm', 'interactions', 'up', 'norm1', 'norm1', 'norm2', 'norm3', 'norm4', 'norm5', 'gcBlock']):
            
            if apply_wd:
                adapter_params['with_wd'].append(param)
            else:
                adapter_params['without_wd'].append(param)
            
        
        elif 'rpn' in name.lower():
            if apply_wd:
                rpn_params['with_wd'].append(param)
            else:
                rpn_params['without_wd'].append(param)
            
        
        elif 'roi_heads' in name.lower() or 'box_head' in name.lower() or 'mask_head' in name.lower():
            if apply_wd:
                roi_head_params['with_wd'].append(param)
            else:
                roi_head_params['without_wd'].append(param)
            
        
        elif 'fpn' in name.lower() or 'neck' in name.lower():
            if apply_wd:
                neck_params['with_wd'].append(param)
            else:
                neck_params['without_wd'].append(param)
            
        else:
            print(name)
            
            if apply_wd:
                roi_head_params['with_wd'].append(param)
            else:
                roi_head_params['without_wd'].append(param)
    
    
    param_groups = []
    
    
    def get_layer_lr(layer_index, base_lr):




        if layer_index == -1:
            
            embedding_lr_factor = lr_decay_rate ** num_vit_layers
            return base_lr * embedding_lr_factor
        else:
            
            dummy_name = f'blocks.{layer_index}'
            lr_factor = lr_decay_func(dummy_name)
            return base_lr * lr_factor
    
    
    embedding_lr = get_layer_lr(-1, base_lr_dinov2)
    
    if embedding_params['with_wd']:
        param_groups.append({
            "params": embedding_params['with_wd'],
            "lr": embedding_lr,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if embedding_params['without_wd']:
        param_groups.append({
            "params": embedding_params['without_wd'],
            "lr": embedding_lr,
            "weight_decay": 0.0,  
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    for i in range(num_vit_layers):
        layer_params = dinov2_layers[f'blocks.{i}']
        layer_lr = get_layer_lr(i, base_lr_dinov2)
        
        if layer_params['with_wd']:
            param_groups.append({
                "params": layer_params['with_wd'],
                "lr": layer_lr,
                "weight_decay": weight_decay,
                "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
            })
        
        if layer_params['without_wd']:
            param_groups.append({
                "params": layer_params['without_wd'],
                "lr": layer_lr,
                "weight_decay": 0.0,
                "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
            })
    
    
    if dinov2_other['with_wd']:
        param_groups.append({
            "params": dinov2_other['with_wd'],
            "lr": base_lr_dinov2,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if dinov2_other['without_wd']:
        param_groups.append({
            "params": dinov2_other['without_wd'],
            "lr": base_lr_dinov2,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    if adapter_params['with_wd']:
        param_groups.append({
            "params": adapter_params['with_wd'],
            "lr": base_lr_adapter,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if adapter_params['without_wd']:
        param_groups.append({
            "params": adapter_params['without_wd'],
            "lr": base_lr_adapter,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    neck_lr = cfg.SOLVER.NECK_LR if hasattr(cfg.SOLVER, 'NECK_LR') else base_lr_adapter
    if neck_params['with_wd']:
        param_groups.append({
            "params": neck_params['with_wd'],
            "lr": neck_lr,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if neck_params['without_wd']:
        param_groups.append({
            "params": neck_params['without_wd'],
            "lr": neck_lr,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    if rpn_params['with_wd']:
        param_groups.append({
            "params": rpn_params['with_wd'],
            "lr": base_lr_rpn,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if rpn_params['without_wd']:
        param_groups.append({
            "params": rpn_params['without_wd'],
            "lr": base_lr_rpn,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    if roi_head_params['with_wd']:
        param_groups.append({
            "params": roi_head_params['with_wd'],
            "lr": base_lr_roi,
            "weight_decay": weight_decay,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    if roi_head_params['without_wd']:
        param_groups.append({
            "params": roi_head_params['without_wd'],
            "lr": base_lr_roi,
            "weight_decay": 0.0,
            "betas": (cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        })
    
    
    print("=" * 80)
    print(" ( -  + bias/norm  weight decay):")
    print(f"   (DINO): {base_lr_dinov2:.2e}")
    print(f"  : {lr_decay_rate}")
    print(f"  ViT : {num_vit_layers}")
    print(f"  : {weight_decay}")
    
    
    num_with_wd = len(embedding_params['with_wd'])
    num_without_wd = len(embedding_params['without_wd'])
    if num_with_wd + num_without_wd > 0:
        embedding_lr = get_layer_lr(-1, base_lr_dinov2)
        print(f"\n  [Embedding] mask_token, cls_token, pos_embed, patch_embed, rope_embed, storage_tokens:")
        print(f"    - with weight decay: {num_with_wd} , lr={embedding_lr:.2e}")
        print(f"    - without weight decay (bias/norm): {num_without_wd} , lr={embedding_lr:.2e}")
    
    
    
    
    print(f"\n  [DINO Blocks] ():")
    for i in range(min(num_vit_layers, 4)):
        layer_params = dinov2_layers[f'blocks.{i}']
        num_with_wd = len(layer_params['with_wd'])
        num_without_wd = len(layer_params['without_wd'])
        if num_with_wd + num_without_wd > 0:
            layer_lr = get_layer_lr(i, base_lr_dinov2)
            print(f"    blocks.{i}: wd={num_with_wd}, no_wd={num_without_wd}, lr={layer_lr:.2e}")
    
    if num_vit_layers > 8:
        print("    ...")
        for i in range(num_vit_layers-4, num_vit_layers):
            layer_params = dinov2_layers[f'blocks.{i}']
            num_with_wd = len(layer_params['with_wd'])
            num_without_wd = len(layer_params['without_wd'])
            if num_with_wd + num_without_wd > 0:
                layer_lr = get_layer_lr(i, base_lr_dinov2)
                print(f"    blocks.{i}: wd={num_with_wd}, no_wd={num_without_wd}, lr={layer_lr:.2e}")
    
    
    print(f"\n  [DINO Other] (norm, head ):")
    print(f"    - with weight decay: {len(dinov2_other['with_wd'])} , lr={base_lr_dinov2:.2e}")
    print(f"    - without weight decay: {len(dinov2_other['without_wd'])} , lr={base_lr_dinov2:.2e}")
    
    print(f"\n  [ViT Adapter]:")
    print(f"    - with weight decay: {len(adapter_params['with_wd'])} , lr={base_lr_adapter:.2e}")
    print(f"    - without weight decay: {len(adapter_params['without_wd'])} , lr={base_lr_adapter:.2e}")
    
    print(f"\n  [Neck]:")
    print(f"    - with weight decay: {len(neck_params['with_wd'])} , lr={neck_lr:.2e}")
    print(f"    - without weight decay: {len(neck_params['without_wd'])} , lr={neck_lr:.2e}")
    
    print(f"\n  [RPN]:")
    print(f"    - with weight decay: {len(rpn_params['with_wd'])} , lr={base_lr_rpn:.2e}")
    print(f"    - without weight decay: {len(rpn_params['without_wd'])} , lr={base_lr_rpn:.2e}")
    
    print(f"\n  [RoI Head]:")
    print(f"    - with weight decay: {len(roi_head_params['with_wd'])} , lr={base_lr_roi:.2e}")
    print(f"    - without weight decay: {len(roi_head_params['without_wd'])} , lr={base_lr_roi:.2e}")
    
    total_groups = len(param_groups)
    print(f"\n  : {total_groups}")
    print("=" * 80)
    
     
    merged_param_groups = reduce_param_groups(param_groups)
    
    print(f": {len(merged_param_groups)}")
    print(f": {len(param_groups) - len(merged_param_groups)} ")
    
    
    optimizer = torch.optim.AdamW(
        merged_param_groups,
        betas=(cfg.SOLVER.BETA1, cfg.SOLVER.BETA2),
        eps=cfg.SOLVER.EPS if hasattr(cfg.SOLVER, 'EPS') else 1e-8,
    )
    
    return optimizer

def build_lr_scheduler_v2(cfg, optimizer):




    
    max_iter = cfg.SOLVER.MAX_ITER
    scheduler_type = cfg.SOLVER.LR_SCHEDULER_NAME
    
    
    
    
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
        lr_multiplier_func = WarmupParamScheduler(
            scheduler=base_scheduler,
            warmup_length=warmup_iters / max_iter,
            warmup_factor=warmup_factor,
            warmup_method=warmup_method,
        )
    else:
        
        
        lr_multiplier_func = base_scheduler
    
    
    
    lr_scheduler = LRMultiplier(
        optimizer=optimizer,
        multiplier=lr_multiplier_func,
        max_iter=max_iter,
        last_iter=-1,  
    )
    
    return lr_scheduler


def add_custom_config(cfg):

    
    cfg.MODEL.BACKBONE.NUM_LAYERS = 12  
    
    
    cfg.SOLVER.OPTIMIZER = "AdamW"
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.WEIGHT_DECAY = 0.1
    cfg.SOLVER.BETA1 = 0.9
    cfg.SOLVER.BETA2 = 0.999
    cfg.SOLVER.EPS = 1e-8
    
    
    cfg.SOLVER.DINOV2_BASE_LR = 1e-4      
    cfg.SOLVER.LR_DECAY_RATE = 0.7        
    cfg.SOLVER.ADAPTER_LR = 1e-4       
    cfg.SOLVER.NECK_LR = 1e-4          
    cfg.SOLVER.RPN_LR = 1e-4           
    cfg.SOLVER.ROI_HEAD_LR = 1e-4         
    
    
    
    
    cfg.SOLVER.USE_FVCORE_SCHEDULER = True  
    cfg.SOLVER.LR_SCHEDULER_NAME = "MultiStep"   
    cfg.SOLVER.STEP_SIZE = 30000
    cfg.SOLVER.MILESTONES = [60000, 80000]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.MIN_LR = 0
    
    
    
    
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 5000
    cfg.SOLVER.WARMUP_METHOD = "linear"   
    
    
    
    
    
