




import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):


    def __init__(
        self,
        apply_norm=True,
        img_level=True,
        remove_neg=True,
        remove_only_teacher_neg=False,
    ):
        super().__init__()

        
        self.mse_loss = torch.nn.MSELoss()

        
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg

    def forward(self, output_feats, target_feats, img_level=True):










        
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3

        
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        
        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)

        if not img_level and len(target_feats.shape) == 3:
            
            target_feats = target_feats.flatten(0, 1)

        
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)

        if not img_level and len(output_feats.shape) == 3:
            
            output_feats = output_feats.flatten(0, 1)

        
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        if self.remove_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        return self.mse_loss(student_sim, target_sim)
