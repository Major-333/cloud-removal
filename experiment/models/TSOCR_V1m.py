import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from models.restormer import LayerNorm, FeedForward, OverlapPatchEmbed, Downsample, Upsample, TransformerBlock


##########################################################################
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads

        self.ms_temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.sar_temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.ms_qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.ms_qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.sar_qkv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.sar_qkv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)


    def forward(self, sar, ms):
        b,c,h,w = ms.shape

        ms_qkv = self.ms_qkv(ms)
        ms_qkv = self.ms_qkv_dwconv(ms_qkv)
        ms_q, ms_k, ms_v = ms_qkv.chunk(3, dim=1)
        ms_q = rearrange(ms_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ms_k = rearrange(ms_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ms_v = rearrange(ms_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ms_q = torch.nn.functional.normalize(ms_q, dim=-1)
        ms_k = torch.nn.functional.normalize(ms_k, dim=-1)


        sar_qkv = self.sar_qkv(sar)
        sar_qkv = self.sar_qkv_dwconv(sar_qkv)
        sar_q, sar_v = sar_qkv.chunk(2, dim=1)
        sar_q = rearrange(sar_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # sar_k = rearrange(sar_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        sar_v = rearrange(sar_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        sar_q = torch.nn.functional.normalize(sar_q, dim=-1)
        # sar_k = torch.nn.functional.normalize(sar_k, dim=-1)

        ms2ms_attn = (ms_q @ ms_k.transpose(-2, -1)) * self.ms_temperature
        ms2ms_attn = ms2ms_attn.softmax(dim=-1)

        sar2ms_attn = (sar_q @ ms_k.transpose(-2, -1)) * self.sar_temperature
        sar2ms_attn = sar2ms_attn.softmax(dim=-1)

        # print(f'ms_v shape:{ms_v.shape}, ms2ms_attn shape:{ms2ms_attn.shape}')
        ms_out = (ms2ms_attn @ ms_v)
        sar_out = (sar2ms_attn @ sar_v)
        
        ms_out = rearrange(ms_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        sar_out = rearrange(sar_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = torch.cat((ms_out, sar_out+ms_out), 1)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerFusionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerFusionBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        sar, inp = x
        inp = inp + self.attn(self.norm1(sar), self.norm1(inp))
        inp = inp + self.ffn(self.norm2(inp))
        return sar, inp



##########################################################################
##---------- Restormer -----------------------
class TSOCR_V1m(nn.Module):
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

        super(TSOCR_V1m, self).__init__()

        self.patch_embed = OverlapPatchEmbed(13, dim)
        self.sar_embed = OverlapPatchEmbed(2, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerFusionBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])
        ])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.sar_down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerFusionBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.sar_down2_3 = Downsample(dim) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerFusionBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.sar_down3_4 = Downsample(dim) ## From Level 3 to Level 4
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
        inp_img = x[:, 2:, :, :]

        inp_enc_level1 = self.patch_embed(inp_img)
        sar_enc_level1 = self.sar_embed(sar)
        _, out_enc_level1 = self.encoder_level1((sar_enc_level1, inp_enc_level1))

        inp_enc_level2 = self.down1_2(out_enc_level1)
        sar_enc_level2 = self.down1_2(sar_enc_level1)
        _, out_enc_level2 = self.encoder_level2((sar_enc_level2, inp_enc_level2))

        inp_enc_level3 = self.down2_3(out_enc_level2)
        sar_enc_level3 = self.down2_3(sar_enc_level2)
        _, out_enc_level3 = self.encoder_level3((sar_enc_level3, inp_enc_level3))

        inp_enc_level4 = self.down3_4(out_enc_level3)    
        sar_enc_level4 = self.down3_4(sar_enc_level3)
        _, latent = self.latent((sar_enc_level4, inp_enc_level4))
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

if __name__ == '__main__':
    fake_input = torch.rand(8, 13, 128, 128) # N C H W
    sar = torch.rand(8, 2, 128, 128) # N C H W
    print(fake_input.shape)
    net = TSOCR_V1m()

    output = net(torch.cat((sar, fake_input), 1))
    print(output.shape)
