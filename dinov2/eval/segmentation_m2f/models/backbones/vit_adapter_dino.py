import torch
from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14, dinov2_vitb14_reg, dinov2_vitl14_reg
from IPython import embed


patch_size, embed_dim, model_func_name = (14, 768, dinov2_vitb14)

img_size = 518
model_name = 'dinov2_vitb14'
path_to_pretrained_weights = "weights/" + model_name + "_pretrain.pth"

encoder = model_func_name(pretrained=False, patch_size=patch_size, img_size=img_size)
encoder.load_state_dict(torch.load(path_to_pretrained_weights),strict=False)

embed()
exit()
