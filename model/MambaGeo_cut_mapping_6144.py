import torch
import torch.nn as nn
from .models_mamba_all_size_cut_mapping_6144 import vim_small_midclstok  # 确保正确导入 Vim 模型
from fvcore.nn import FlopCountAnalysis, flop_count_table

class MambaGeo(nn.Module):
    """
    A Siamese model wrapper for Vim to handle two different input images.
    """
    def __init__(self, args, pretrained_path=None):
        super(MambaGeo, self).__init__()
        self.dim = args.dim

        if args.dataset == 'vigor':
            self.size_sat = [320, 320]
            self.size_sat_default = [320, 320]
            self.size_grd = [320, 640]
        elif args.dataset == 'cvusa':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [128, 512]
        elif args.dataset == 'cvact':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [128, 512]

        # 根据 FOV 设定地面图像宽度
        if args.fov != 0:
            self.size_grd[1] = int(args.fov / 360.0 * self.size_grd[1])

        # 实例化模型时传入图像大小
        # self.query_net = vim_small_midclstok(img_size=self.size_grd, patch_size=(32, 8), stride=(32, 8))
        self.query_net = vim_small_midclstok(img_size=self.size_grd)
        self.reference_net = vim_small_midclstok(img_size=self.size_sat)
        self.polar = None

    def forward(self, im_q, im_k):
        out_q = self.query_net(im_q)
        out_k = self.reference_net(im_k)
        return out_q, out_k

# class Args:
#     def __init__(self, dataset='cvact', fov=0, dim=3072):
#         self.dataset = dataset
#         self.fov = fov
#         self.dim = dim

# # 创建一个包装类，将 frgeo_model 包装在其中
# class WrappedFRGeo(nn.Module):
#     def __init__(self, model):
#         super(WrappedFRGeo, self).__init__()
#         self.model = model

#     def forward(self, im_q, im_k):
#         return self.model(im_q, im_k)

# # 设置模型参数并实例化模型
# args = Args(dataset='cvusa', fov=0, dim=3072)
# frgeo_model = MambaGeo(args)
# wrapped_model = WrappedFRGeo(frgeo_model)

# # 创建符合输入尺寸的假输入数据
# images_q = torch.randn(1, 3, 128, 512)  # 地面图像尺寸
# images_k = torch.randn(1, 3, 256, 256)  # 卫星图像尺寸

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# wrapped_model = wrapped_model.to(device)
# images_q = images_q.to(device)
# images_k = images_k.to(device)

# # 计算 FLOPs 和参数
# with torch.no_grad():  # 禁用梯度计算以加速推理
#     flops = FlopCountAnalysis(wrapped_model, (images_q, images_k))
#     print(flop_count_table(flops))  # 输出 FLOPs 和参数表

# # 验证模型输出
# output_q, output_k = wrapped_model(images_q, images_k)
# print("Output shape of out_q:", output_q.shape)
# print("Output shape of out_k:", output_k.shape)
