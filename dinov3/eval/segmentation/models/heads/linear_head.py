




import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):


    def __init__(
        self,
        in_channels,
        n_output_channels,
        use_batchnorm=True,
        use_cls_token=False,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        if use_cls_token:
            self.channels *= 2  
        self.n_output_channels = n_output_channels
        self.use_cls_token = use_cls_token
        self.batchnorm_layer = nn.SyncBatchNorm(self.channels) if use_batchnorm else nn.Identity(self.channels)
        self.conv = nn.Conv2d(self.channels, self.n_output_channels, kernel_size=1, padding=0, stride=1)
        self.dropout = nn.Dropout2d(dropout)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def _transform_inputs(self, inputs):






        inputs = [
            torch.nn.functional.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for x in inputs
        ]
        inputs = torch.cat(inputs, dim=1)
        return inputs

    def _forward_feature(self, inputs):








        
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if self.use_cls_token:
                assert len(x) == 2, "Missing class tokens"
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs):

        output = self._forward_feature(inputs)
        output = self.dropout(output)
        output = self.batchnorm_layer(output)
        output = self.conv(output)
        return output

    def predict(self, x, rescale_to=(512, 512)):





        x = self._forward_feature(x)
        x = self.batchnorm_layer(x)
        x = self.conv(x)
        x = F.interpolate(input=x, size=rescale_to, mode="bilinear")
        return x
