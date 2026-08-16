import argparse
import numpy as np
import torch
import os
import lpips
from piq import ssim, psnr
from place2.real_CMYW_GT_PatchDataset import CMYW_RGB_PatchDataset
from torch.utils.data import DataLoader
from src import mymodels, summary_utils, utils
from torchvision.utils import save_image
import time

if torch.cuda.device_count() == 5:
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count() == 4:
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count() == 3:
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
elif torch.cuda.device_count() == 2:
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
utils.seed(num=123, deterministic=True)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='../data/split/test')
    parser.add_argument('--test_epoch', type=str, default='.1_best_epoch.')
    parser.add_argument('--m', type=int, help='表示要读取的第几行数据', default=1)
    parser.add_argument('--snapshot', type=str,
                        default='../snapshots/default')
    parser.add_argument('--save_image_dir', type=str,
                        default='../image_result', help='保存图片路径')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--bool_noise', type=bool, default=False)
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('-b', '--block_size',
                        help='delimited list input for block size in format(格式中块大小的分隔列表输入) %,%,%',
                        default=[4, 256, 256])
    parser.add_argument('--cfa_size', help='cfa的尺寸大小,只针对lcmyw有效%,%,%',
                        default=[4, 4])
    parser.add_argument('--interp', type=str, choices=['none',
                                                       'gaussian'],
                        default='none')  # 原版required=True, default='pconv'/
    parser.add_argument('--init', type=str, choices=['softmax_tau',
                                                     'gumbel_softmax_tau_1',
                                                     'gumbel_softmax_tau_2'],
                        default='gumbel_softmax_tau_2',
                        help='针对可学习CMYW CFA的形式,反向传播代理函数问题,'
                             'softmax_tau:带温度系数的softmax函数,gama=2.5e-5,可去函数里面调整,tau根据迭代次数进行变化,'
                             'gumbel_softmax_tau_1:Gumbel_Softmax函数,tau为初始值1,原始GB-ST'
                             'gumbel_softmax_tau_2:Gumbel_Softmax,tau可变,根据迭代次数变化,噪声项含有独立温度系数')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--decoder', type=str,
                        choices=['unet', 'RDNet', 'dncnn', 'myunet', 'Uformer',
                                 'mysgnet', 'drunet', 'rlfn', 'nafnet',
                                 'mysgnet1', 'rrdbnet', 'rdunet', 'simdunet',
                                 'RestNet', 'RASTNet', 'DGAsimdunet', 'DGAGLMIFunet',
                                 'Res3unet', 'JDENet', 'DNF', 'DUJRENet'],
                        default='JDENet')  # 原版default='unet'
    parser.add_argument('--shutter', type=str, default='lcmyw')
    args = parser.parse_args()

    read_root_txt = '{:s}/ckpt/{:s}/{:s}/set_canshu.txt'.format(args.snapshot, args.shutter, args.interp)
    fo_main = open(read_root_txt, 'r')
    read_lines = fo_main.readlines()
    print('read_lines', read_lines)
    print('len(read_lines)', len(read_lines))
    print("要读取的数据为第{}行".format(args.m))
    m = args.m
    cfa_size = []
    parameter = read_lines[m].strip('\n').split(',')
    args.alpha = float(parameter[3])
    args.w_zhi = float(parameter[4])
    args.gaussian_weight_size = int(parameter[5])
    cfa_sz = parameter[7].split('x')
    cfa_size.append(int(cfa_sz[0]))
    cfa_size.append(int(cfa_sz[1]))
    args.cfa_size = cfa_size

    snapshot = '{:s}/ckpt/{:s}/{:s}/{:s}/{:s}_epoch.pth'.format(args.snapshot,
                                                                 args.shutter,
                                                                 args.interp,
                                                                 args.init,
                                                                 args.test_epoch)
    print('snapshot', snapshot)

    # 自采数据集测试集（从 .npy 加载）
    test_cmyw_path = os.path.join(args.root, "CMYW_data.npy")
    test_gt_path = os.path.join(args.root, "GT_data.npy")
    dataset_test = CMYW_RGB_PatchDataset(
        test_cmyw_path,
        test_gt_path,
        transform=None
    )
    print('len(dataset_test)', len(dataset_test))

    test_dataloader = DataLoader(dataset_test,
                                 batch_size=1,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 )

    shutter = mymodels.define_myshutter(args.shutter, args)
    decoder = mymodels.define_mydecoder(args.decoder, args)
    model = mymodels.define_mymodel(shutter, decoder, args, get_coded=False)

    checkpoint = torch.load(snapshot, map_location='cuda')
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    with torch.no_grad():
        psnr_list = []
        ssim_list = []
        lpips_list = []
        for step, (model_input, gt) in enumerate(test_dataloader, 1):
            model_input = model_input.to(device)
            gt = gt.to(device)
            deblur, cfa_img = model(model_input, train=False)
            deblur = torch.clamp(deblur, 0, 1)
            gt_rgb = gt[:, :3, :, :]

            # 计算指标
            psnr = summary_utils.get_psnr(deblur, gt_rgb)
            print(f"第{step}张图片PSNR：{psnr:.4f}")
            psnr_list.append(psnr)

            ssim_val = ssim(deblur, gt_rgb, data_range=1.0)
            ssim_scalar = ssim_val.mean().item()
            print(f"第{step}张图片SSIM：{ssim_scalar:.4f}")
            ssim_list.append(ssim_scalar)

            deblur_norm = deblur * 2 - 1
            gt_norm = gt_rgb * 2 - 1
            lpips_val = lpips_model(deblur_norm, gt_norm)
            lpips_scalar = lpips_val.mean().item()
            print(f"第{step}张图片LPIPS：{lpips_scalar:.4f}")
            lpips_list.append(lpips_scalar)

            # ---- 保存图像 ----
            save_dir = os.path.join(args.save_image_dir, args.shutter, args.interp, args.init)
            os.makedirs(save_dir, exist_ok=True)

            # 保存重建结果
            save_path = os.path.join(save_dir, f"result_{step:04d}.png")
            save_image(deblur, save_path)

            # 保存对应的真值图像（GT RGB）
            gt_path = os.path.join(save_dir, f"gt_{step:04d}.png")
            save_image(gt_rgb, gt_path)

            # ====== 已移除 CFA 图像的保存 ======

        # 统计结果
        print('psnr_list', psnr_list)
        mean_psnr = np.mean(psnr_list)
        mean_ssim = np.mean(ssim_list)
        mean_lpips = np.mean(lpips_list)

        print(f'Mean PSNR : {mean_psnr:.4f}')
        print(f'Mean SSIM : {mean_ssim:.4f}')
        print(f'Mean LPIPS: {mean_lpips:.4f}')

    print("block_size:", args.block_size)
    print("cfa_size:", args.cfa_size)
    print("shutter: ", args.shutter)
    print("interpolation: ", args.interp)
    print("decoder:", args.decoder)
    print('代理函数：', args.init)
    print('固定cfa初始化值为：', args.w_zhi)
    print("alpha: ", args.alpha)
    print("gaussian_weight_size: ", args.gaussian_weight_size)
    print("是否加噪声:", args.bool_noise)
    print("噪声等级std:", args.noise_std)
    end_time = time.time()
    print(f"程序运行时间：{end_time - start_time:.4f}s")