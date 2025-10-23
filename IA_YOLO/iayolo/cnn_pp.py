import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_PP(nn.Module):
    """
    CNN-PP Module PyTorch Reproduction
    """
    def __init__(self, input_dim=3, output_dim=15):
        super(CNN_PP, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_PP(output_dim=15).to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    out = model(x)
    print(out.shape)  # [1, 15]
    summary(model, input_size=(3, 256, 256))  # 约 165K
