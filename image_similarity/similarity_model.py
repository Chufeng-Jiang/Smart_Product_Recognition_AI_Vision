__all__ = ["ConvEncoder", "ConvDecoder"]

import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
    def forward(self, x):
        return self.model(x)

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    x = torch.randn(1, 3, 64, 64)
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    encoded_x = encoder(x)
    decoded_x = decoder(encoded_x)
    print(encoded_x.shape)
    print(decoded_x.shape)
