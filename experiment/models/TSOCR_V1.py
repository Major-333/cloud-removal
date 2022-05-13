## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from models.restormer import LayerNorm, FeedForward, OverlapPatchEmbed, Downsample, Upsample, TransformerBlock


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.ms_temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.ms_qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.ms_qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)

        self.sar_temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.sar_qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.sar_qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)

        self.ms_project_out = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)
        self.sar_project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, sar, ms):
        b,c,h,w = ms.shape

        ms_q, ms_k, ms_v = self.ms_qkv_dwconv(self.ms_qkv(ms)).chunk(3, dim=1)     
        
        ms_q = rearrange(ms_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ms_k = rearrange(ms_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ms_v = rearrange(ms_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        ms_q = torch.nn.functional.normalize(ms_q, dim=-1)
        ms_k = torch.nn.functional.normalize(ms_k, dim=-1)

        sar_q, sar_k, sar_v = self.sar_qkv_dwconv(self.sar_qkv(sar)).chunk(3, dim=1)     
        
        sar_q = rearrange(sar_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        sar_k = rearrange(sar_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        sar_v = rearrange(sar_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        sar_q = torch.nn.functional.normalize(sar_q, dim=-1)
        sar_k = torch.nn.functional.normalize(sar_k, dim=-1)

        ms_self_attn = (ms_q @ ms_k.transpose(-2, -1)) * self.ms_temperature
        ms_self_attn = ms_self_attn.softmax(dim=-1)

        ms_self_attn_out = (ms_self_attn @ ms_v)
        
        ms_self_attn_out = rearrange(ms_self_attn_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        sar_self_attn = (sar_q @ sar_k.transpose(-2, -1)) * self.sar_temperature
        sar_self_attn = sar_self_attn.softmax(dim=-1)

        sar_self_attn_out = (sar_self_attn @ sar_v)
        
        sar_self_attn_out = rearrange(sar_self_attn_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        cross_attn = (sar_q @ ms_k.transpose(-2, -1)) * self.ms_temperature
        cross_attn = cross_attn.softmax(dim=-1)

        cross_attn_out = (cross_attn @ sar_v)

        cross_attn_out = rearrange(cross_attn_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        sar_out = self.sar_project_out(sar_self_attn_out)
        ms_out = self.ms_project_out(torch.cat((ms_self_attn_out, cross_attn_out + ms_self_attn_out), 1))
        ms_out = ms + ms_out # short cut
        sar_out = sar + sar_out # short cut

        return sar_out, ms_out



##########################################################################
class TransformerFusionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerFusionBlock, self).__init__()
        self.ms_norm1 = LayerNorm(dim, LayerNorm_type)
        self.sar_norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)
        self.ms_norm2 = LayerNorm(dim, LayerNorm_type)
        self.sar_norm2 = LayerNorm(dim, LayerNorm_type)
        self.ms_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.sar_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        sar, ms = x
        sar, ms = self.attn(self.sar_norm1(sar), self.ms_norm1(ms))

        sar = sar + self.sar_ffn(self.sar_norm2(sar))
        ms = ms + self.ms_ffn(self.ms_norm2(ms))

        return (sar, ms)


class TSOCR_V1(nn.Module):
    def __init__(self, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(TSOCR_V1, self).__init__()

        self.ms_patch_embed = OverlapPatchEmbed(13, dim)
        self.sar_patch_embed = OverlapPatchEmbed(2, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerFusionBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.ms_down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.sar_down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[TransformerFusionBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.ms_down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.sar_down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[TransformerFusionBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.ms_down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.sar_down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4

        self.latent = nn.Sequential(*[TransformerFusionBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), 13, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        sar = x[:, :2, :, :]
        ms = x[:, 2:, :, :]

        sar_encode_level1 = self.sar_patch_embed(sar)
        ms_encode_level1 = self.ms_patch_embed(ms)
        sar_encode_level1_out, ms_encode_level1_out = self.encoder_level1((sar_encode_level1, ms_encode_level1))

        sar_encode_level2 = self.sar_down1_2(sar_encode_level1_out)
        ms_encode_level2 = self.ms_down1_2(ms_encode_level1_out)
        sar_encode_level2_out, ms_encode_level2_out = self.encoder_level2((sar_encode_level2, ms_encode_level2))

        sar_encode_level3 = self.sar_down2_3(sar_encode_level2_out)
        ms_encode_level3 = self.ms_down2_3(ms_encode_level2_out)
        sar_encode_level3_out, ms_encode_level3_out = self.encoder_level3((sar_encode_level3, ms_encode_level3))

        sar_encode_level4 = self.sar_down3_4(sar_encode_level3_out)
        ms_encode_level4 = self.ms_down3_4(ms_encode_level3_out)
        _, latent = self.latent((sar_encode_level4, ms_encode_level4))
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, ms_encode_level3_out], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, ms_encode_level2_out], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, ms_encode_level1_out], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(ms_encode_level1_out)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + ms


        return out_dec_level1

if __name__ == '__main__':
    fake_input = torch.rand(8, 13, 128, 128) # N C H W
    sar = torch.rand(8, 2, 128, 128) # N C H W
    print(fake_input.shape)
    net = TSOCR_V1()

    output = net(torch.cat((sar, fake_input), 1))
    print(output.shape)
