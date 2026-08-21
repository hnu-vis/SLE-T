




import torch
from mmcv.ops import point_sample


def get_uncertainty(mask_pred, labels):

















    if mask_pred.shape[1] == 1:
        gt_class_logits = mask_pred.clone()
    else:
        inds = torch.arange(mask_pred.shape[0], device=mask_pred.device)
        gt_class_logits = mask_pred[inds, labels].unsqueeze(1)
    return -torch.abs(gt_class_logits)


def get_uncertain_point_coords_with_randomness(
    mask_pred, labels, num_points, oversample_ratio, importance_sample_ratio
):






















    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_pred.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(batch_size, num_sampled, 2, device=mask_pred.device)
    point_logits = point_sample(mask_pred, point_coords)
    
    
    
    
    
    
    
    
    
    point_uncertainties = get_uncertainty(point_logits, labels)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(batch_size, dtype=torch.long, device=mask_pred.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(batch_size, num_uncertain_points, 2)
    if num_random_points > 0:
        rand_roi_coords = torch.rand(batch_size, num_random_points, 2, device=mask_pred.device)
        point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
    return point_coords
