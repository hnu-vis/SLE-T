import torch
from torch import nn

import torch.nn.init as init

import torch.nn.functional as F

from IPython import embed

def last_zero_init_raw(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[-1].weight, 0.0)
        if m[-1].bias is not None:
            nn.init.constant_(m[-1].bias, 0.0)
    else:
        nn.init.constant_(m.weight, 0.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def init_projection_mask(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            
            init.ones_(m.weight)
            init.zeros_(m.bias)

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 out_channels,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.out_channels = out_channels
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.BatchNorm2d(self.planes),
                nn.ReLU(inplace=True),  
                nn.Conv2d(self.planes, self.out_channels, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.BatchNorm2d(self.planes),
                nn.ReLU(inplace=True),  
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
            
        self.projection_mask = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes // 2, kernel_size=1),
            nn.BatchNorm2d(self.inplanes // 2),
            
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // 2, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            
            nn.ReLU(inplace=True),
        )
        
        self.relu = nn.ReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            torch.nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in', nonlinearity='relu')
            init_projection_mask(self.projection_mask)
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init_raw(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init_raw(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            
            input_x = input_x.view(batch, channel, height * width)
            
            input_x = input_x.unsqueeze(1)
            
            context_mask = self.conv_mask(x)
            
            context_mask = context_mask.view(batch, 1, height * width)
            
            context_mask = self.softmax(context_mask)
            
            context_mask = context_mask.unsqueeze(-1)
            
            context = torch.matmul(input_x, context_mask)
            
            context = context.view(batch, channel, 1, 1)
        else:
            
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        
        context = self.spatial_pool(x)

        out = self.projection_mask(x)
        if self.channel_mul_conv is not None:
            
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
            
            
            out = self.relu(out)

        return out
    



class ContextBlock_stride(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 out_channels,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock_stride, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.out_channels = out_channels
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.BatchNorm2d(self.planes),
                nn.ReLU(inplace=True),  
                nn.Conv2d(self.planes, self.out_channels, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.BatchNorm2d(self.planes),
                nn.ReLU(inplace=True),  
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
            
        self.projection_mask = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes // 2, kernel_size=1),
            nn.BatchNorm2d(self.inplanes // 2),
            
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // 2, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.relu = nn.ReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            torch.nn.init.kaiming_normal_(self.conv_mask.weight, mode='fan_in', nonlinearity='relu')
            init_projection_mask(self.projection_mask)
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init_raw(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init_raw(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            
            input_x = input_x.view(batch, channel, height * width)
            
            input_x = input_x.unsqueeze(1)
            
            context_mask = self.conv_mask(x)
            
            context_mask = context_mask.view(batch, 1, height * width)
            
            context_mask = self.softmax(context_mask)
            
            context_mask = context_mask.unsqueeze(-1)
            
            context = torch.matmul(input_x, context_mask)
            
            context = context.view(batch, channel, 1, 1)
        else:
            
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        
        context = self.spatial_pool(x)

        out = self.projection_mask(x)
        if self.channel_mul_conv is not None:
            
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
            
            
            out = self.relu(out)

        return out


class ContextBlock_lt(nn.Module):
    def __init__(self,
                 inplanes,
                 out_channels,):
        super(ContextBlock_lt, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(inplanes, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
        self.reset_parameters()
        
    def reset_parameters(self):
        
        for module in self.projection.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        out = self.projection(x)
        
        return out
    