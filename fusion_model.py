import torch
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn as nn
import geffnet






class AutoFusion(nn.Module):
    """docstring for AutoFusion"""
    def __init__(self ,in_ch,latent_dim):
        super(AutoFusion, self).__init__()
        self.in_ch = in_ch
        self.latent_dim = latent_dim

        self.fuse_in = nn.Sequential(
            nn.Linear(in_ch,in_ch//2),
            nn.Tanh(),
            nn.Linear(in_ch//2,latent_dim),
            nn.ReLU()
            )
        self.fuse_out = nn.Sequential(
            nn.Linear(latent_dim, in_ch//2),
            nn.ReLU(),
            nn.Linear(in_ch//2,in_ch)
            )
        self.criterion = nn.MSELoss()

    def forward(self, z):
        compressed_z = self.fuse_in(z)
        loss = self.criterion(self.fuse_out(compressed_z), z)
        output = {
            'z': compressed_z,
            'loss': loss
        }
        return output
        
        
sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
        


class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False,fusion = True,latent_dim = 1024):
        super(Effnet_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.classifier.in_features
        #print("inchannel before concat",in_ch)
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
            if fusion == True :
                self.fusion_model = AutoFusion(in_ch,latent_dim)
        #print("inchannel after concat",in_ch)
        self.out = nn.Linear(latent_dim, out_dim)
        self.enet.classifier = nn.Identity()

    def fuse(self, z_dict):
        """
        Input:
        ======
        z_dict for AutoFusion and no fusion will have only one latent code:
        the concatenation of all z from all modalities
        z_dict for GanFusion will have the individual modalities z separated
        as we'll need to define a GanFusion module for every modality
        Returns:
        ========
        an output dictionary with logits and aux_loss
        """
        if isinstance(self.fusion_model, AutoFusion):
            fusion_output_dict =  self.fusion_model(z_dict['z'])
            z_fuse = fusion_output_dict['z']
            aux_loss = fusion_output_dict['loss']
            return {
                'logits': self.out(z_fuse),
                'aux_loss': aux_loss
            }
        else:
            raise NotImplementedError

    def extract(self, x):
        x = self.enet(x)
        
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        #print("extracted shape",x.shape)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            #print("meta shape",x_meta.shape)
            concat_x = torch.cat((x, x_meta), dim=1)
            #print("x + x_mata shape",concat_x.shape)
        return self.fuse({'z': concat_x})

