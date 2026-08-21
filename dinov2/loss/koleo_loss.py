




import logging

import torch
import torch.nn as nn
import torch.nn.functional as F




logger = logging.getLogger("dinov2")


class KoLeoLoss(nn.Module):


    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):




        
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  
        
        _, I = torch.max(dots, dim=1)  
        return I

    def forward(self, student_output, eps=1e-8):




        with torch.cuda.amp.autocast(enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  
            distances = self.pdist(student_output, student_output[I])  
            loss = -torch.log(distances + eps).mean()
        return loss
