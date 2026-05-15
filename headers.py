# This headers replace PartialFC

import torch
import math
from torch import nn
import torch.nn.functional as F

  
        

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, num_classes, embedding_size, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size) * 0.01)
        self.num_classes = num_classes 
        
        self.s = s
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        n_weight = F.normalize(self.weight)
        n_feats = F.normalize(feats)
        
        logits = torch.mm(n_feats, n_weight.T)
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, num_classes, embedding_size, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.weight = nn.Parameters(torch.randn(num_classes, embedding_size) * 0.01)
        self.num_classes = num_classes 
                
        self.s = s
        self.m = m

    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        n_weight = F.normalize(self.weight)
        n_feats = F.normalize(feats)        
        logits = torch.mm(n_feats, n_weight.T)
        
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits

    
class AdaCos(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size) * 0.01)
        self.s =  math.log2((num_classes - 1)) * math.sqrt(2)
        
    def forward(self, x, labels):
        weight = F.normalize(self.weight)
        x = F.normalize(x)
        logits = F.linear(x, weight)
        with torch.no_grad():
            index = torch.where(labels != -1)[0]
            B_avg = torch.exp(self.s * logits)
            B_avg[index, labels[index].view(-1)] = 0
            B_avg = B_avg.sum() / logits.size(0)
            target_logit = logits[index, labels[index].view(-1)]
            target_theta = torch.acos(target_logit.clamp(-1, 1))
            theta_med = torch.median(target_theta)
            self.s = torch.log(B_avg + 1e-6) / (torch.cos(torch.min(torch.pi/4 * torch.ones_like(theta_med), theta_med)) + 1e-6)
        
        return self.s * logits  
    
    
class ECosFace(nn.Module):
    def __init__(self, num_classes, embedding_size, s=64.0, m=0.35, std=0.025, plus=True):
        super().__init__()
        self.num_classes = num_classes
        self.emb_size = embedding_size
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size) * 0.01)
        self.std=std
        self.plus=plus

    def forward(self, x, labels):
        weight = F.normalize(self.weight)
        x = F.normalize(x)
        logits = F.linear(x, weight)
        index = torch.where(labels != -1)[0]
        m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device)
        margin = torch.normal(mean=self.m, std=self.std, size=labels[index, None].size(), device=logits.device)  # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = logits[index, labels.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, labels[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, labels[index, None], margin)
        logits[index] -= m_hot
        ret = logits * self.s
        return ret       
        
class AdaFace(nn.Module):
    def __init__(self, num_classes, embedding_size, s = 64, m = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size) * 0.01)
        self.m = m
        self.s = s
        self.h = 0.333
        
        self.bn = nn.BatchNorm1d(1, affine = False, track_running_stats = True, momentum=0.01)
    
    @torch.no_grad()
    def feature_QA(self, x_norm):
        x_hat = self.bn(x_norm) * self.h
        x_hat = x_hat.clamp(-1, 1)
        return x_hat
        
        
    def forward(self, x, labels):
        nx = F.normalize(x)
        weight = F.normalize(self.weight)
        logits = F.linear(nx, weight, None)
        
        with torch.no_grad():
            x_norm = x.norm(p=2,dim=1, keepdim = True)
            x_hat = self.feature_QA(x_norm)
            g_ang = -self.m * x_hat
            g_add = self.m * x_hat + self.m
        
        with torch.no_grad():
            index = torch.where(labels != -1)[0]
            target_logit = logits[index, labels[index].view(-1)]
            target_theta = torch.acos(target_logit)            
            final_target_logit = target_theta.view(x.size(0), 1) + g_ang
            logits[index, labels[index].view(-1)] = (torch.cos(final_target_logit) - g_add).squeeze()
            
        return self.s * logits
