import sys
import os
import shutters.myshutters as myshutters
# from nets.unet import UNet
from nets.rlfn import RLFN
from nets.NAFNet_arch import NAFNet

from nets.rrdb import RRDBNet
# from nets.mysgnet1 import MySgNet1
from nets.mysgnet import MySGNet
from nets.rdunet import RDUNet
from src.myTreeMultiRandom import MyTreeScatter
from PConv.PConv import PartialConv2d,PartialConv2d_Not_Official
from nets.unet_myself import UNetMyself
import torch.nn.functional as F
from nets.sgnet import NET
from nets.shufflemixer_arch import ShuffleMixer
# from nets.msanet import MSANet
from nets.RestNet import RESTCANet
from nets.network_unet import *
# from DNFmodels.dnf_model import DNF

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() ==5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count()==3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def define_myshutter(shutter_type,args,test=False):
    return myshutters.Shutter(block_size=args.block_size,shutter_type=shutter_type,cfa_size=args.cfa_size,
                w_zhi=args.w_zhi,alpha=args.alpha,test=test,init=args.init)

def define_mymodel(shutter, decoder, args, get_coded=False):
    #在编码器和解码器之间是否定义插值模块
    if args.interp=='gaussian' and args.shutter=='lcmyw':
        print('*******添加LCMYW-Gaussian插值*********')
        return GaussianModel(shutter=shutter,decoder=decoder,gaussian_weight_size=args.gaussian_weight_size)
    elif args.interp=='none' and args.shutter=='lcmyw':
        print('*****不添加插值模块*******')
        return Model(shutter=shutter,decoder=decoder,get_coded=get_coded)

    raise NotImplementedError('Interp + Shutter combo has not been implemented')

def define_mydecoder(model_name, args):
    if args.decoder == 'none':
        return None
    elif args.shutter=='lrgbw' and args.interp=='gaussian':
        in_ch = 4
        out_ch = 3
    elif args.shutter=='lrgbw' and args.interp=='none':
        in_ch=4
        out_ch=3
    else:
        raise NotImplementedError
    if model_name=='myunet':
        return UNetMyself(n_channels=in_ch,n_classes=out_ch)
    elif model_name == 'unet':
        #对于RGBW必须要加BatchNormal层，否则没有效果
        return UNet(in_nc=in_ch, out_nc=out_ch, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose') #upsample 原版up_mode='upconv' batch_norm=False
    elif model_name=='sgnet':
        return NET(sr_n_resblocks=6,dm_n_resblocks=6,channels=24,scale=2,denoise=False,
                block_type='rrdb',act_type='relu',bias=False,norm_type=None)
    elif model_name=='mysgnet':
        return MySGNet(in_nc=3, out_nc=3,
                 nc=[64, 128, 256, 512], nb=2, act_mode='R',
                 downsample_mode="strideconv", upsample_mode="convtranspose")
    elif model_name=='simdunet':
        return SIMDUNet(in_nc=3,out_nc=3,nc=[64, 128, 256, 512], nb=3, act_mode='R',
                 downsample_mode="strideconv", upsample_mode="convtranspose")
    elif model_name=='rlfn':
        return RLFN(in_channels=4,out_channels=3,feature_channels=52)
    elif model_name=='nafnet':
        middle_blk_num=24
        enc_blks = [2, 2, 4, 8]
        dec_blks = [2, 2, 2, 2]
        return NAFNet(img_channel=4, width=32, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    elif model_name == 'Uformer':
        return Uformer(f_number=64, block_size=2, layers=4).cuda()
    elif model_name == 'RDNet':
        return RDNet(f_number=64, block_size=2, layers=4).cuda()
    elif model_name=='rrdbnet':
        return RRDBNet(channels=64,act_type='leakyrelu',bias=True,norm_type=None)
    elif model_name=='rdunet':
        return RDUNet(inchannels=4,outchannels=3,base_filters=64) #原版base_filters=64，可学习测试显存爆炸
    elif model_name=='shufflemixer':
        return ShuffleMixer(n_feats=64, kernel_size=7, n_blocks=5, mlp_ratio=2, upscaling_factor=1)
    elif model_name =='msanet':
        return MSANet(input_channel=4, output_channel=3)
    elif model_name=='unetres':
        return UNetRes(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=2, act_mode='R', 
                 downsample_mode="strideconv", upsample_mode="convtranspose")
    elif model_name=='RestNet':
        return RESTCANet(in_nc=4, out_nc=3, patch_size=2, nc=72,
                         window_size=8,
                         num_heads=6,
                         depths=2)
    elif model_name=='RASTNet':
        return RASTNet (nc=[64, 128, 256, 512])

    elif model_name=='DGAsimdunet':
        return DGASIMDUNet(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=3, act_mode='R',
                        downsample_mode="strideconv", upsample_mode="convtranspose")
    elif model_name=='DGAGLMIFunet':
        return DGAGLMIFUNet(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R',
                        downsample_mode="strideconv", upsample_mode="pixelshuffle")
    elif model_name=='Res3unet':
        return Res3UNet(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=3, act_mode='R',
                 downsample_mode="strideconv", upsample_mode="pixelshuffle")
    elif model_name=='JDENet':
        return JDENet(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=3, act_mode='G',
                 downsample_mode="strideconv", upsample_mode="convtranspose")
    elif model_name=='DNF':
        return DNF(f_number=64, block_size=2, layers=4).cuda()
    elif model_name=='DUJRENet':
        return DUJRENet(in_nc=4,out_nc=3,nc=[64, 128, 256, 512], nb=3, act_mode='R',
                 downsample_mode="strideconv", upsample_mode="pixelshuffle")
    raise NotImplementedError('Model not specified correctly')






class Model(nn.Module):
    def __init__(self, shutter, decoder,  get_coded=False):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        self.decoder = decoder

    def forward(self, input, train=True):
        # print('self.shutter',self.shutter)
        # print('input.shape',input.shape)#torch.Size([1, 4, 150, 150])
        coded,mask= self.shutter(input, train=train)
        # print('coded.shape',coded.shape)
        x = self.decoder(coded)
        if self.get_coded:
            return x, coded
        return x, mask

class GaussianModel(nn.Module):
    def __init__(self, shutter, decoder,gaussian_weight_size):
        super(GaussianModel, self).__init__()
        self.shutter = shutter
        self.decoder = decoder
        self.Gaussian_weight_size=gaussian_weight_size
        # print(self.Gaussian_weight_size)
        # exit()
        # self.Gaussian_weight=torch.normal(mean=0,std=9/6,size=(4,1,9,9),device=device)
        weight=torch.normal(mean=0,std=(self.Gaussian_weight_size/(self.Gaussian_weight_size-2)),size=(self.Gaussian_weight_size,self.Gaussian_weight_size),device=device)
        # print('weight',weight)
        # exit()
        weight=weight[None,None,:,:]
        self.Gaussian_weight=torch.cat((weight,weight,weight,weight),dim=0)

    def forward(self, image,train=True):
        # print('self.Gaussian_weight',self.Gaussian_weight)
        rgbw_raw, mask = self.shutter(image,train=train)
        # save_image(rgbw_raw[:,:3,:,:],'../image_result/cfa_img.png')
        mask_4=mask[None,:,:,:]
        # print('rgbw_raw.shape',rgbw_raw.shape)
        # print('mask_4.shape',mask_4.shape) #torch.Size([1, 4, 256, 256])
        # print('rgbw_raw',rgbw_raw)
        rgbw_Gaussian=F.conv2d(input=rgbw_raw,weight=self.Gaussian_weight,stride=1,
                                padding=self.Gaussian_weight_size//2,groups=4,bias=None)
        mask_Gaussian=F.conv2d(input=mask_4,weight=self.Gaussian_weight,stride=1,
                                padding=self.Gaussian_weight_size//2,groups=4,bias=None)
        epsilon=0.1/255
        rgbw=rgbw_Gaussian/(mask_Gaussian+epsilon)
        # 把原来的像素值赋值回去
        # print(rgbw_raw)
        # index = torch.where(rgbw_raw != 0,1,2) #torch1.1.0版本
        # index=torch.nonzero(rgbw_raw)
        # print(index)
        # print(len(index))
        index = torch.where(rgbw_raw != 0)
        # print(index)
        # exit()
        data = rgbw_raw
        temp = data[index]
        rgbw[index] = temp
        # print(rgbw == rgbw_raw)
        # exit()
        # print(rgbw_Gaussian==0)
        # print(mask_Gaussian==0)
        # print(rgbw==0)
        # print('rgbw',rgbw)
        # exit()
        # print(rgbw==0)
        # save_image(rgbw[:,:3,:,:],'../image_result/gaussian_img.png')
        # exit()
        output=self.decoder(rgbw)

        # output=self.decoder(rgbw_Gaussian)
        # output.register_hook(print)
        return output,mask
    

