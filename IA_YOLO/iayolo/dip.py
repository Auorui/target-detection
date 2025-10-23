import torch
import torch.nn as nn
import torch.nn.functional as F
from IA_YOLO.iayolo.cnn_pp import CNN_PP


def rgb2lum(image):
    image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :, 1] + 0.06 * image[:, :, :, 2]
    return image[:, :, :, None]


def lerp(a, b, l):
    return (1 - l) * a + l * b


def tanh01(x):
    return (torch.tanh(x) + 1) / 2


def tanh_range(l, r, initial=None):
    def get_activation(left, right, initial):
        def activation(x):
            if initial is not None:
                bias = torch.atanh(2 * (initial - left) / (right - left) - 1)
            else:
                bias = 0
            return tanh01(x + bias) * (right - left) + left

        return activation

    return get_activation(l, r, initial)


class Dip(nn.Module):
    def __init__(self):
        super(Dip, self).__init__()
        self.t0 = 0.1
        self.top_percent = 0.1  # 用于估计大气光的像素百分比
        self.cnn_pp = CNN_PP(input_dim=3, output_dim=15)
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())

    def _create_gaussian_kernel(self, sigma=5, device='cuda:0'):
        """创建高斯核"""
        radius = 12
        x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        k = torch.exp(-0.5 * torch.square(x / sigma))
        k = k / torch.sum(k)
        kernel_2d = k.unsqueeze(1) * k.unsqueeze(0)  # [25, 25]
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, 25, 25]
        kernel_2d = kernel_2d.repeat(3, 1, 1, 1)  # [3, 1, 25, 25]
        return kernel_2d

    def _apply_white_balance(self, image, param):
        """白平衡滤镜"""
        param = param * 1.0 / (0.27 * param[:, 0] +
                               0.67 * param[:, 1] + 0.06 * param[:, 2] + 1e-5)[:, None]
        return image * param[:, :, None, None]

    def _apply_defog(self, x, param):
        # https://blog.csdn.net/m0_62919535/article/details/148291799
        # 输入形状: (B, C, H, W)，值域[0,1]
        if x.min() < 0:  # 检测到输入是[-1,1]范围
            x = (x + 1) / 2  # 转换到[0,1]
        dark = self.dark_channel(x)
        A = self.estimate_atmosphere(x, dark)
        if A.shape[1] != x.shape[1]:
            A = A[:, :x.shape[1], :, :]
        transmission = self.transmission(dark, param)
        if transmission.dim() == 3:
            transmission = transmission.unsqueeze(1)
        # 根据物理模型恢复图像
        J = (x - A) / transmission + A
        return torch.clamp(J, 0, 1)

    def _apply_gamma(self, image, param):
        param_1 = param.repeat([1, 3])
        return torch.pow(torch.max(image, torch.tensor(0.0001)), param_1[:, :, None, None])

    def _apply_tone(self, image, param):
        tone_curve = param
        tone_curve_sum = torch.sum(tone_curve, dim=-1) + 1e-30
        total_image = image * 0
        curve_steps = 8
        for i in range(curve_steps):
            total_image = total_image + torch.clip(image - 1.0 * i / curve_steps, 0, 1.0 / curve_steps) \
                          * param[:, :, :, :, i]
        total_image = total_image * curve_steps / tone_curve_sum
        return total_image

    def _apply_contrast(self, image, param):
        luminance = torch.min(torch.max(rgb2lum(image), torch.tensor(0.0)), torch.tensor(1.0))
        contrast_lum = -torch.cos(torch.pi * luminance) * 0.5 + 0.5
        contrast_image = image / (luminance + 1e-6) * contrast_lum
        return lerp(image, contrast_image, param[:, :, None, None])

    def _apply_sharpen(self, image, param):
        """锐化处理 - 修复高斯核创建问题"""
        device = image.device
        image = image.float()
        kernel_i = self._create_gaussian_kernel(5, device)

        # 使用卷积进行高斯模糊
        output = F.conv2d(image, weight=kernel_i, stride=1, groups=3, padding=12)
        img_out = (image - output) * param[:, None, None, :] + image
        return img_out

    def forward(self, x):
        params = self.cnn_pp(x)
        params = self._parse_cnn_params(params)
        image = x
        B, C, W, H = x.shape
        # 图像处理流程
        image = self._apply_defog(image, params['defog_omega'])
        image = self._apply_white_balance(image, params['white_balance'])
        image = self._apply_gamma(image, params['gamma_correction'])
        image = self._apply_tone(image, params['tone_curve_points'])
        image = self._apply_contrast(image, params['contrast_adjust'])
        image = self._apply_sharpen(image, params['sharpen_strength'])
        result_image = torch.sigmoid(image)
        return result_image

    def _parse_cnn_params(self, params):
        """解析 CNN_PP 输出的参数并应用约束"""
        batch_size = params.shape[0]

        # dark channel
        omega = tanh_range(*(0.1, 1.0))(params[:, 0:1])

        # wb
        wb_mask = torch.tensor(((0, 1, 1)), device=params.device).unsqueeze(0)
        features = params[:, 1:4] * wb_mask
        wb = torch.exp(tanh_range(-0.5, 0.5)(features))

        # gamma
        gamma_range_val = torch.tensor(3.0, device=params.device)
        log_gamma_range = torch.log(gamma_range_val)
        gamma = torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(params[:, 4:5]))

        # tone
        tone_curve = torch.reshape(params[:, 5:13], shape=(-1, 1, 8))[:, None, None, :]
        tone_curve = tanh_range(*(0.5, 2))(tone_curve)

        # Contrast
        contrast = torch.tanh(params[:, 13:14])

        # sharpen
        sharpen = tanh_range(*(0.0, 5))(params[:, 14:15])

        params_dict = {
            'defog_omega': omega,
            'white_balance': wb,
            'gamma_correction': gamma,
            'tone_curve_points': tone_curve,
            'contrast_adjust': contrast,
            'sharpen_strength': sharpen
        }
        return params_dict

    def dark_channel(self, img):
        """计算暗通道 (B, C, H, W) -> (B, H, W)"""
        return torch.min(img, dim=1)[0]  # 取RGB通道最小值

    def estimate_atmosphere(self, img, dark_ch):
        """估计大气光A"""
        B, H, W = dark_ch.shape
        # 选择暗通道中前0.1%最亮的像素
        num_pixels = int(H * W * self.top_percent)
        flattened_dark = dark_ch.view(B, -1)
        indices = torch.topk(flattened_dark, num_pixels, dim=1)[1]
        # 获取原始图像中对应位置的像素
        atmosphere = []
        for b in range(B):
            selected_pixels = img[b, :, indices[b] // W, indices[b] % W]
            atmosphere.append(torch.max(selected_pixels, dim=1)[0])
        return torch.stack(atmosphere).unsqueeze(-1).unsqueeze(-1)

    def transmission(self, dark_ch, omega):
        """计算透射率图"""
        if omega.ndim == 2:
            omega = omega.view(-1, 1, 1)
        elif omega.ndim == 1:
            omega = omega.view(-1, 1, 1)
        transmission = 1 - omega * dark_ch
        return torch.clamp(transmission, min=self.t0, max=1.0)


if __name__ == "__main__":
    import cv2
    import numpy as np

    dip = Dip()
    haze = cv2.resize(cv2.imread(r'E:\PythonProject\target_detection\IA_YOLO\data\images\XR_Baidu_375.png'), (608, 608))
    image_tensor = torch.from_numpy(
        cv2.cvtColor(haze, cv2.COLOR_BGR2RGB)
    ).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    tensor = dip(image_tensor)
    image = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    cv2.imshow("enhance", np.hstack([haze, image[..., ::-1]]))
    cv2.waitKey(0)