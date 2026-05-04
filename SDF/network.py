import torch
import numpy as np
import os
import sys
import permutohedral_encoding as permuto_enc
import torch.nn as nn


class SDF(torch.nn.Module):

    def __init__(self, in_channels, geom_feat_size_out, nr_iters_for_c2f):
        super(SDF, self).__init__()

        self.in_channels=in_channels
        self.geom_feat_size_out=geom_feat_size_out


        #create encoding
        pos_dim=in_channels
        capacity=pow(2,18) #2pow18
        nr_levels=24 
        nr_feat_per_level=2 
        coarsest_scale=1.0 
        finest_scale=0.0001 
        scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
        self.encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, appply_random_shift_per_level=True, concat_points=True, concat_points_scaling=1e-3)           

        
        self.sdf_shift=1e-2
        self.mlp_sdf= torch.nn.Sequential(
            torch.nn.Linear(self.encoding.output_dims() ,32),
            torch.nn.GELU(),
            torch.nn.Linear(32,32),
            torch.nn.GELU(),
            torch.nn.Linear(32,32),
            torch.nn.GELU(),
            torch.nn.Linear(32,1+geom_feat_size_out)
        )
        apply_weight_init_fn(self.mlp_sdf, leaky_relu_init, negative_slope=0.0)
        leaky_relu_init(self.mlp_sdf[-1], negative_slope=1.0)
        with torch.set_grad_enabled(False):
            self.mlp_sdf[-1].bias+=self.sdf_shift #faster if we just put it in the bias

        # self.mlp_sdf=torch.compile(self.mlp_sdf, mode="max-autotune")

       


        self.c2f=permuto_enc.Coarse2Fine(nr_levels)
        self.nr_iters_for_c2f=nr_iters_for_c2f
        self.last_iter_nr=sys.maxsize

    def forward(self, points, iter_nr=20000):

        assert points.shape[1] == self.in_channels, "points should be N x in_channels"

        self.last_iter_nr=iter_nr

       
        window=self.c2f( map_range_val(iter_nr, 0.0, self.nr_iters_for_c2f, 0.3, 1.0   ) )

     
        point_features=self.encoding(points, window.view(-1))
        sdf_and_feat=self.mlp_sdf(point_features)
        
        if self.geom_feat_size_out!=0:
            sdf=sdf_and_feat[:,0:1]
            geom_feat=sdf_and_feat[:,-self.geom_feat_size_out:]
        else:
            sdf=sdf_and_feat
            geom_feat=None


        return sdf, geom_feat

    def get_sdf_and_gradient(self, points, iter_nr=20000, method="autograd"):


        if method=="finite_difference":
            with torch.set_grad_enabled(False):
                #to the original positions, add also a tiny epsilon in all directions
                nr_points_original=points.shape[0]
                epsilon=1e-4
                points_xplus=points.clone()
                points_yplus=points.clone()
                points_zplus=points.clone()
                points_xplus[:,0]=points_xplus[:,0]+epsilon
                points_yplus[:,1]=points_yplus[:,1]+epsilon
                points_zplus[:,2]=points_zplus[:,2]+epsilon
                points_full=torch.cat([points, points_xplus, points_yplus, points_zplus],0)

               
            sdf_full, geom_feat_full = self.forward(points_full, iter_nr)

            geom_feat=None
            if geom_feat_full is not None:            
                g_feats=geom_feat_full.chunk(4, dim=0) 
                geom_feat=g_feats[0]

            sdfs=sdf_full.chunk(4, dim=0) 
            sdf=sdfs[0]
            sdf_xplus=sdfs[1]
            sdf_yplus=sdfs[2]
            sdf_zplus=sdfs[3]

            grad_x=(sdf_xplus-sdf)/epsilon
            grad_y=(sdf_yplus-sdf)/epsilon
            grad_z=(sdf_zplus-sdf)/epsilon

            gradients=torch.cat([grad_x, grad_y, grad_z],1)


        elif method=="autograd":

            #do it with autograd
            with torch.set_grad_enabled(True):
                points.requires_grad_(True)
                sdf, geom_feat = self.forward(points, iter_nr)

                feature_vectors=None
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]



        return sdf, gradients, geom_feat



def leaky_relu_init(m, negative_slope=0.2):

    gain = np.sqrt(2.0 / (1.0 + negative_slope ** 2))

    if isinstance(m, torch.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return

  
    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, torch.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

def apply_weight_init_fn(m, fn, negative_slope=1.0):

    should_initialize_weight=True
    if not hasattr(m, "weights_initialized"): #if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    elif m.weights_initialized==False: #if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    else:
        print("skipping weight init on ", m)
        should_initialize_weight=False

    if should_initialize_weight:
        # fn(m, is_linear, scale)
        fn(m,negative_slope)
        # m.weights_initialized=True
        for module in m.children():
            apply_weight_init_fn(module, fn, negative_slope)


def map_range_val( input_val, input_start, input_end,  output_start,  output_end):
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    input_clamped=max(input_start, min(input_end, input_val))
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)




class mlp_feature(nn.Module):
    def __init__(self, view_dim, level_dim, feat_dim):
        super(mlp_feature, self).__init__()
        self.mlp_feature_bank = nn.Sequential(
                    nn.Linear(view_dim+level_dim, feat_dim),
                    nn.ReLU(True),
                    nn.Linear(feat_dim, 3),
                    nn.Softmax(dim=1)
                ).cuda()
    
    def forward(self,input):
        return self.mlp_feature_bank(input)
            

class mlp_opacity(nn.Module):
    def __init__(self,feat_dim, view_dim, opacity_dist_dim,level_dim,n_offsets):
        super(mlp_opacity, self).__init__()

        self.mlp_opac = nn.Sequential(
                nn.Linear(feat_dim+view_dim+opacity_dist_dim+level_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, n_offsets),
                nn.Tanh()
            ).cuda()
    
    def forward(self,input):
        return self.mlp_opac(input)

class mlp_roughness(nn.Module):
    def __init__(self, feat_dim, view_dim, opacity_dist_dim,level_dim,n_offsets):
        super(mlp_roughness, self).__init__()

        self.mlp_roug = nn.Sequential(
                nn.Linear(feat_dim+view_dim+opacity_dist_dim+level_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, n_offsets),
                nn.Sigmoid()
            ).cuda()
    
    def forward(self,input):
        return self.mlp_roug(input)



class mlp_matallic(nn.Module):
    def __init__(self, feat_dim, view_dim, opacity_dist_dim,level_dim,n_offsets):
        super(mlp_matallic, self).__init__()

        self.mlp_mata = nn.Sequential(
                nn.Linear(feat_dim+view_dim+opacity_dist_dim+level_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, n_offsets),
                nn.Sigmoid()
            ).cuda()
    
    def forward(self,input):
        return self.mlp_mata(input)
      

class mlp_cov(nn.Module):
    def __init__(self, feat_dim,view_dim, level_dim, cov_dist_dim,n_offsets):
        super(mlp_cov, self).__init__()

        self.mlp_covT = nn.Sequential(
                nn.Linear(feat_dim+view_dim+cov_dist_dim+level_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 7*n_offsets),
            ).cuda()
    
    def forward(self,input):
        return self.mlp_covT(input)
    

class mlp_color(nn.Module):
    def __init__(self,  feat_dim,view_dim, color_dist_dim,level_dim,appearance_dim,n_offsets):
        super(mlp_color, self).__init__()

        self.mlp_colorT = nn.Sequential(
                nn.Linear(feat_dim+view_dim+color_dist_dim+level_dim+appearance_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*n_offsets),
                nn.Sigmoid()
            ).cuda()
    
    def forward(self,input):
        return self.mlp_colorT(input)


class mlp_albedo(nn.Module):
    def __init__(self, feat_dim,view_dim, color_dist_dim,level_dim,appearance_dim,n_offsets):
        super(mlp_albedo, self).__init__()

        self.mlp_albe = nn.Sequential(
                nn.Linear(feat_dim+view_dim+color_dist_dim+level_dim+appearance_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*n_offsets),
                nn.Sigmoid()
            ).cuda()
    
    def forward(self,input):
        return self.mlp_albe(input)
       
class mlp_normal1(nn.Module):
    def __init__(self, feat_dim,view_dim, color_dist_dim,level_dim,appearance_dim,n_offsets):
        super(mlp_normal1, self).__init__()

        self.mlp_norm1= nn.Sequential(
                nn.Linear(feat_dim+view_dim+color_dist_dim+level_dim+appearance_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*n_offsets),
                nn.Sigmoid()
            ).cuda()
    
    def forward(self,input):
        return self.mlp_norm1(input)

class mlp_normal2(nn.Module):
    def __init__(self, feat_dim,view_dim, color_dist_dim,level_dim,appearance_dim,n_offsets):
        super(mlp_normal2, self).__init__()

        self.mlp_norm2= nn.Sequential(
                nn.Linear(feat_dim+view_dim+color_dist_dim+level_dim+appearance_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*n_offsets),
                nn.Sigmoid()
            ).cuda()
    
    def forward(self,input):
        return self.mlp_norm2(input)



