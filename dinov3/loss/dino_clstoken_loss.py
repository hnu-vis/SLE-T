




import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from dinov3.distributed import get_process_subgroup, get_subgroup_size


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.full((1, out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    def init_weights(self) -> None:
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        
        teacher_output = teacher_output.float()
        world_size = get_subgroup_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  
        B = Q.shape[1] * world_size  
        K = Q.shape[0]  

        
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q, group=get_process_subgroup())
        Q /= sum_Q

        for _ in range(n_iterations):
            
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows, group=get_process_subgroup())
            Q /= sum_of_rows
            Q /= K

            
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  
        return Q.t()

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
















        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        student_logits = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        if not ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> ", student_logits, teacher_probs)
            return loss / (B * student_crops * teacher_crops)
        else:
            loss = -torch.einsum("s b k, t b k -> s t", student_logits, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            return loss.sum() / (B * student_crops * teacher_crops - B * min_st)

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True, group=get_process_subgroup())

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = get_subgroup_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
