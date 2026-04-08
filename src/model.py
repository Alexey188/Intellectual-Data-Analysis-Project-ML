import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # First convolution to extract features
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # Stabilizes training / Стабилизирует обучение
            nn.ReLU(inplace=True),

            # Second convolution to refine features
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# U-Net: Encoder-Decoder architecture with skip connections
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Downsampling via MaxPool / Уменьшение размерности через MaxPool
        self.pool = nn.MaxPool2d(2)

        # Encoder (Downsampling path)
        # Энкодер (путь сжатия)
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)

        # deepest feature representation
        # самое глубокое представление признаков
        self.bottleneck = DoubleConv(256, 512)

        # Decoder (Upsampling path)
        self.up1 = nn.ConvTranspose2d(512, 256, 2,2)
        self.up_conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.up_conv4 = DoubleConv(64, 32)

        # Final 1x1 convolution to produce the binary mask
        # Финальная свертка 1x1 для формирования бинарной маски
        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # Forward through Encoder and save intermediate results for skip connections
        # Проход через энкодер с сохранением промежуточных результатов для skip connections
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        bn = self.bottleneck(self.pool(d4))

        # Forward through Decoder using concatenation (Skip connections)
        # Проход через декодер с использованием конкатенации (Skip connections)
        # torch.cat combines deep semantic features with spatial details from encoder
        # torch.cat объединяет глубокие семантические признаки с деталями из энкодера
        u1 = self.up_conv1(torch.cat([self.up1(bn), d4], dim=1))
        u2 = self.up_conv2(torch.cat([self.up2(u1), d3], dim=1))
        u3 = self.up_conv3(torch.cat([self.up3(u2), d2], dim=1))
        u4 = self.up_conv4(torch.cat([self.up4(u3), d1], dim=1))

        return self.out_conv(u4)