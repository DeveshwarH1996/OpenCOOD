import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import gc


class PillarCrossAttention(nn.Module):
    def __init__(self, model_cfg, num_pillar_features):
        '''Pillar Cross Attention Module
            It is a module that takes the input features of the pillars and applies the cross attention mechanism,
            relative to the input features of the pillars themselves. It applies a kernel like operation to the input
            features of the pillars and computes the cross attention relative to the neighboring pillars
        Args:
            model_cfg (dict): Dictionary containing the configuration of the model
            num_pillar_features (int): Number of input features of the pillars
            num_output_CA_features (int): Number of additional output features of the Pillar Cross Attention Module
        '''
        
        super().__init__()
        self.model_cfg = model_cfg
        self.num_pillar_features = num_pillar_features

        self.num_output_CA_features = self.model_cfg['num_output_CA_features']
        self.kernel_size = self.model_cfg['kernel_size']

        # Linear layers for query, key and value
        self.linear_query = nn.Linear(self.num_pillar_features, self.num_output_CA_features)
        self.linear_key = nn.Linear(self.num_pillar_features, self.num_output_CA_features)
        self.linear_value = nn.Linear(self.num_pillar_features, self.num_output_CA_features)

        # Sinusoidal positional encoding
        self.positional_encoding = self.sinusoidal_positional_encoding(self.kernel_size, self.num_pillar_features)
        self.positional_encoding = nn.Parameter(self.positional_encoding, requires_grad=False)
        self.is_mask = False

    def sinusoidal_positional_encoding(self, kernel_size, num_pillar_features):
        '''Sinusoidal Positional Encoding
            It generates the sinusoidal positional encoding of shape (kernel_size**2, num_pillar_features)
        Args:
            kernel_size (int): Kernel size of the cross attention mechanism
            num_pillar_features (int): Number of input features of the pillars
        Returns:
            torch.Tensor: Sinusoidal positional encoding
        '''
        
        # Generate the positions
        positions = torch.arange(0, kernel_size**2, dtype=torch.float32).unsqueeze(1)
        # Generate the dimensions
        dimensions = torch.arange(0, num_pillar_features, 2, dtype=torch.float32)

        # Compute the angles
        angles = positions / torch.pow(10000, 2 * dimensions / num_pillar_features)

        # Compute the sinusoidal positional encoding
        positional_encoding = torch.zeros(kernel_size**2, num_pillar_features)
        positional_encoding[:, 0::2] = torch.sin(angles)
        positional_encoding[:, 1::2] = torch.cos(angles)

        return positional_encoding
    
    def create_mask(self, H, W, kernel_size):
        '''Create Mask
            It creates a mask for the cross attention mechanism
        Args:
            H (int): Height of the mask
            W (int): Width of the mask
            kernel_size (int): Kernel size of the cross attention mechanism
        Returns:
            torch.Tensor: Mask
        '''
        
        mask = torch.ones(H, W).unsqueeze(0).unsqueeze(1)
        mask = F.unfold(mask, (kernel_size, kernel_size), stride=1)
        mask = F.fold(mask, (H, W), (kernel_size, kernel_size), stride=1)
        mask = mask.detach()
        self.is_mask = True

        return mask


    def forward(self, batch_dict):
        '''Forward pass of the Pillar Cross Attention Module
        '''
        pseudo_image = batch_dict['spatial_features']
        B, C, H, W = pseudo_image.shape
        cross_attention = torch.zeros(B, self.num_output_CA_features, H, W).to(pseudo_image.device)
        self.cross_attention_map = self.create_mask(H, W, self.kernel_size).to(pseudo_image.device) if not self.is_mask else self.cross_attention_map

        length_quotient = H // self.kernel_size
        length_remainder = H % self.kernel_size
        is_reduce_length = False
        width_quotient = W // self.kernel_size
        width_remainder = W % self.kernel_size
        is_reduce_width = False

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                
                # Extract the kernel
                if i > length_remainder and not is_reduce_length:
                    length_quotient -= 1
                    is_reduce_length = True
                
                if j > width_remainder and not is_reduce_width:
                    width_quotient -= 1
                    is_reduce_width = True
                
                kernel = pseudo_image[:, :, i:i+self.kernel_size*length_quotient, j:j+self.kernel_size*width_quotient].to(pseudo_image)
                kernel = einops.rearrange(kernel, 'b c (l k1) (w k2) -> b (l w) (k1 k2) c', k1 = self.kernel_size, k2 = self.kernel_size)

                # Compute the positional encoding
                kernel += self.positional_encoding.to(pseudo_image.device)

                # Compute the query, key and value
                query = self.linear_query(kernel)
                key = self.linear_key(kernel)
                value = self.linear_value(kernel)


                # Compute the attention
                attention = torch.matmul(query, key.transpose(2, 3))
                attention = F.softmax(attention, dim=-1)

                # Compute the output
                attention = torch.matmul(attention, value).to(pseudo_image.device)
                attention = einops.rearrange(attention, 'b (l w) (k1 k2) c -> b c (l k1) (w k2)', l=length_quotient, k1 = self.kernel_size)

                # Update the pseudo image
                cross_attention[:, :, i:i+self.kernel_size*length_quotient, j:j+self.kernel_size*width_quotient] += attention
        
        # Normalize the pseudo image
        cross_attention /= self.cross_attention_map

        # Concatenate the cross attention with the pseudo image
        batch_dict['spatial_features'] = torch.cat([pseudo_image, cross_attention], dim=1)

        return batch_dict
    

    # def forward(self, batch_dict):
    #     '''Forward pass of the Pillar Cross Attention Module
    #     '''
    #     torch.cuda.empty_cache()
    #     kernel = batch_dict['spatial_features']
    #     B, C, H, W = kernel.shape

    #     kernel = F.unfold(kernel, (self.kernel_size, self.kernel_size), stride=1).to(kernel.device)
    #     kernel = kernel.permute(0, 2, 1).reshape(B, -1, self.kernel_size * self.kernel_size)
    #     kernel = einops.rearrange(kernel,'b (no_kernels d) k -> b no_kernels k d', d=self.num_pillar_features)
    #     kernel += self.positional_encoding.to(kernel.device)

    #     query = self.linear_query(kernel)
    #     key = self.linear_key(kernel)
    #     value = self.linear_value(kernel)

    #     kernel = torch.matmul(query, key.transpose(2, 3))
    #     kernel = F.softmax(kernel, dim=-1)

    #     del query
    #     del key
    #     gc.collect()

    #     kernel = torch.matmul(kernel, value)

    #     del value
    #     gc.collect()

    #     kernel = kernel.view(B, -1, self.num_output_CA_features, self.kernel_size * self.kernel_size)
    #     kernel = einops.rearrange(kernel, 'b no_kernels d k -> b no_kernels (d k)', d=self.num_output_CA_features).permute(0, 2, 1)
    #     kernel = F.fold(kernel, (H, W), (self.kernel_size, self.kernel_size), stride=1) 

    #     if not self.is_mask:
    #         self.cross_attention_mask = self.create_mask(H, W, self.kernel_size).to(kernel.device)
    #         self.is_mask = True
        
    #     kernel /= self.cross_attention_mask

        

    #     # Concatenate the cross attention with the pseudo image
    #     batch_dict['spatial_features'] = torch.cat([batch_dict['spatial_features'], kernel], dim=1)

    #     return batch_dict