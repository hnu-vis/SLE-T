




import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F

import dinov3.distributed as dist


class KoLeoLoss(nn.Module):


    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):




        
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  
        _, indices = torch.max(dots, dim=1)  
        return indices

    def forward(self, student_output, eps=1e-8):




        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            indices = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[indices])  
            loss = -torch.log(distances + eps).mean()
        return loss


class KoLeoLossDistributed(nn.Module):


    def __init__(self, topk=1, loss_group_size: int | None = None):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        self.topk = topk
        self.loss_group_size = loss_group_size  

    def pairwise_NNs_inner(self, x, all_x, rank):




        
        dots = torch.mm(x, all_x.t())  
        local_B, global_B = dots.shape
        dots.view(-1)[rank * local_B :: (global_B + 1)].fill_(-1)  
        _, indices = torch.topk(dots, dim=1, k=self.topk)  
        return indices

    def forward(self, student_output, eps=1e-8):




        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)  

            if dist.is_enabled():
                all_student_outputs = torch.cat(torch_dist.nn.all_gather(student_output), dim=0)  
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                all_student_outputs = student_output
                world_size = 1
                rank = 0

            
            
            local_B = len(student_output)
            global_B = len(all_student_outputs)
            loss_group_size = self.loss_group_size if self.loss_group_size is not None else global_B
            if loss_group_size % local_B != 0:
                raise ValueError(
                    f"Loss group size size {loss_group_size} must be a multiple of local batch size {local_B}."
                )
            if global_B % loss_group_size != 0:
                raise ValueError(
                    f"Global batch size {global_B} must be divisible by loss group size {loss_group_size}."
                )
            n_groups = global_B // loss_group_size
            ranks_per_group = world_size // n_groups
            rank_in_group = rank % ranks_per_group
            group = rank // ranks_per_group
            all_student_outputs = all_student_outputs.view(n_groups, loss_group_size, student_output.shape[1])
            all_student_outputs = all_student_outputs[group]  

            with torch.no_grad():
                indices = self.pairwise_NNs_inner(student_output, all_student_outputs, rank_in_group)  

            student_output_expanded = (
                student_output.unsqueeze(1).repeat(1, self.topk, 1).flatten(0, 1)
            )  
            distances = self.pdist(student_output_expanded, all_student_outputs[indices].flatten(0, 1))  
            loss = -torch.log(distances.float() + eps).mean()

        return loss
