import torch
import cv2
import numpy as np
from IA_YOLO.iayolo import ConvWithIA
from IA_YOLO.models.yolo import Model

if __name__=="__main__":
    yaml_path = r"E:\PythonProject\target_detection\IA_YOLO\models\iayolov5n.yaml"
    weight_path = r"E:\PythonProject\target_detection\IA_YOLO\runs\train\exp3\weights\best.pt"

    model = Model(yaml_path)
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"].float().state_dict(), strict=False)
    model.eval()

    first_module = model.model[0]
    if isinstance(first_module, ConvWithIA):
        print("Found ConvWithIA module")
    else:
        print(f"Warning: The first module is {type(first_module)}")

    img_path = r"E:\PythonProject\target_detection\data\RTTS_split\train\images\XR_Baidu_375.png"

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (512, 512))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        out, enhanced = first_module(img_tensor, return_enhanced=True)

    print(f"Input shape    : {img_tensor.shape}")
    print(f"Enhanced shape : {enhanced.shape}")
    print(f"Output shape   : {out.shape}")

    enhanced_np = torch.clamp(enhanced, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_bgr = cv2.cvtColor((enhanced_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    original_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    compare = np.hstack((original_bgr, enhanced_bgr))
    cv2.imshow("Original (Left) vs Enhanced (Right)", compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()