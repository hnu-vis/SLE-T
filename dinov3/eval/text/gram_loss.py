




import torch
import torch.nn.functional as F


def gram_loss_fn(
    backbone_patch_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    patch_sampling_rate: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    num_patches, dim = patch_tokens.shape[1:]
    idx = torch.randperm(num_patches)[: int(num_patches * patch_sampling_rate)]
    patch_tokens = patch_tokens[:, idx, :]
    backbone_patch_tokens = backbone_patch_tokens[:, idx, :]
    if normalize:
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        backbone_patch_tokens = F.normalize(backbone_patch_tokens, dim=-1)
    return torch.nn.MSELoss()(
        patch_tokens @ patch_tokens.transpose(-2, -1),
        backbone_patch_tokens @ backbone_patch_tokens.transpose(-2, -1),
    )
