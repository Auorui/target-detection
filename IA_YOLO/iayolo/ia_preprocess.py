import torch
import torch.nn as nn
from IA_YOLO.iayolo.dip import Dip

class ConvWithIA(nn.Module):
    """
    将 IA_Preprocess 功能集成到第一个 Conv。
    输入: RGB 3通道
    输出: 原 Conv 输出通道数
    """
    def __init__(self, in_channels=3, out_channels=64, kernel_size=6, stride=2, padding=2):
        super().__init__()
        # 前置增强模块
        self.dip = Dip()
        # 原 Conv 模块
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x, return_enhanced=False):
        # IA_Preprocess
        enhanced = self.dip(x)
        enhanced = enhanced.to(self.conv.weight.dtype)
        # 原 Conv
        out = self.conv(enhanced)
        out = self.bn(out)
        out = self.act(out)
        if return_enhanced:
            return out, enhanced
        return out

if __name__=="__main__":
    import numpy as np
    # x = torch.randn(1, 3, 512, 512)
    # model = ConvWithIA()
    # with torch.no_grad():
    #     out = model(x)
    # print("Input shape: ", x.shape)
    # print("Output shape:", out.shape)
    from IA_YOLO.models.yolo import Model

    # n/s/m/l/x
    model = Model(r"E:\PythonProject\target_detection\IA_YOLO\models\iayolov5n.yaml")
    model.info(verbose=True)

    # YOLOv5n summary: 214 layers, 1872157 parameters, 1872157 gradients, 4.6 GFLOPs
    # iaYOLOv5n summary: 232 layers, 2037116 parameters, 2037116 gradients, 35.3 GFLOPs

    # yaml_path = r"E:\PythonProject\target_detection\IA_YOLO\models\iayolov5n.yaml"
    # weight_path = r"E:\PythonProject\target_detection\IA_YOLO\runs\train\exp13\weights\best.pt"
    #
    # full_model = Model(yaml_path)
    # ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    # full_model.load_state_dict(ckpt["model"].float().state_dict(), strict=False)
    # full_model.eval()
    # print("✅ 模型加载完成")
    # import cv2
    # # =====================================
    # # 2️⃣ 提取第一个 ConvWithIA 模块
    # # =====================================
    # # 假设第一个模块在 full_model.model[0]
    # first_module = full_model.model[0]
    # if isinstance(first_module, ConvWithIA):
    #     print("✅ 成功找到 ConvWithIA 模块")
    # else:
    #     print(f"⚠️ 注意：第一个模块类型是 {type(first_module)}")
    #
    # # =====================================
    # # 3️⃣ 读取测试图像
    # # =====================================
    # img_path = r"E:\PythonProject\target_detection\data\RTTS_split\train\images\BD_Baidu_095.png"
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_resized = cv2.resize(img, (512, 512))
    # img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    #
    # # 前向传播并取增强结果
    # with torch.no_grad():
    #     out, enhanced = first_module(img_tensor, return_enhanced=True)
    #
    # print("Input shape :", img_tensor.shape)
    # print("Enhanced shape:", enhanced.shape)
    # print("Output shape :", out.shape)
    #
    # # 显示增强效果
    # enhanced_np = torch.clamp(enhanced, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    # enhanced_bgr = cv2.cvtColor((enhanced_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    # original_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    #
    # compare = np.hstack((original_bgr, enhanced_bgr))
    # cv2.imshow("Original (Left) vs Enhanced (Right)", compare)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()