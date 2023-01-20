import FrEIA.framework as Ff
import FrEIA.modules as Fm

import torch
import torch.nn as nn
import torch.nn.functional as F

import constant as const

class AltUB(nn.Module):
    '''
    Implementation of
    Kim, Y. et al. (2022). AltUB: Alternating training method to update 
    base distribution of normalizing flow for anomaly detection.

    As we mentioned in discussion 4.6, the usage of AltUB is optional.
    AltUB trains the base distribution N(mu, sigma^2). 
    Otherwise, the base distribution is fixed as N(0, 1).
    '''

    def __init__(self, dim):
        super().__init__()
        self.base_mean = nn.Parameter(torch.zeros(1, dim))
        self.base_cov = nn.Parameter(torch.zeros(1, dim))

    def forward(self, z, rev=False):
        if rev:
            return z
        z = (z-self.base_mean)/torch.exp(self.base_cov)
        return z

#Subnet for the coupling blocks
def subnet_fc(dims_in, dims_out):
    hidden_channels = int(dims_in * const.RATIO_FC)
    return nn.Sequential(nn.Linear(dims_in, hidden_channels), nn.ReLU(),
                         nn.Linear(hidden_channels,  dims_out))

#Subnet for the conditional affine coupling layer
def subnet_cat(dims_in, dims_out):
    hidden_channels = int(round(dims_in * const.RATIO_CAT))
    return nn.Sequential(nn.Linear(dims_in, hidden_channels), nn.GELU(),
                         nn.Linear(hidden_channels,  dims_out))

#Construction of normalizing flow 
#Conditional normalizing flow & coupling block
def nf_flow(dim, flow_steps, cond_dims):
    nodes = Ff.SequenceINN(dim)
    nodes.append(
            Fm.ConditionalAffineTransform, #Conditional affine transformation
            cond = 0,
            cond_shape=(cond_dims, ),
            subnet_constructor = subnet_cat,
            dims_c = [(cond_dims,)],
            clamp_activation = "TANH",
            clamp = 2.25
        )
    for i in range(flow_steps):    
        nodes.append(
            Fm.AllInOneBlock,
            cond = 0,
            cond_shape = (cond_dims, ),
            subnet_constructor = subnet_fc,
            permute_soft = True, 
            gin_block = True, #We used GIN coupling layer
            #learned_householder_permutation = 1 #Hyperparameter option
            #reverse_permutation = True #Hyperparameter option
        )

    return nodes


#The main model, CD_Flow (Flow model for Chronic Diseases prediction)
class CD_Flow(nn.Module):
    def __init__(self, dim_features, flow_steps, cond_dims):
        super().__init__()
        self.nf_flows = nf_flow(dim_features, flow_steps, cond_dims)
        self.base = AltUB(dim_features) #Base distribution
        
    def forward(self, x, c, rev=False):
        loss = 0
        ret = {}
        output, log_jac_dets = self.nf_flows(x, [c, ], rev=rev)
        output = self.base(output, rev=rev)

        #Loss implementation. Note that the base distribution is assumed to be N(mu, sigma^2) when using AltUB.
        loss += torch.mean(
            torch.sum(0.5*(output**2)+self.base.base_cov) - log_jac_dets
        )

        ret['loss'] = loss
        ret['output'] = output

        if not self.training:
            log_prob = -torch.mean(0.5*(output**2) +self.base.base_cov , dim=1)
            prob = torch.exp(log_prob)
            #To calculate an anomaly score, the negative sign is taken to the likelihood. 
            #The higher score, the more likely to be an anomaly.
            ret['score'] = -prob

        return ret