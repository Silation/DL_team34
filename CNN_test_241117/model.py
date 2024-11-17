import torch
import torch.nn as nn

DEBUG = 0

class GoodEdgeDistributionCNN(nn.Module):
    def __init__(self):
        super(GoodEdgeDistributionCNN, self).__init__()
        
        # Encoding layers (Downsampling)
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  # k5
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=6, stride=2, padding=2)  # k6
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # k5
        self.enc_conv4 = nn.Conv2d(64, 64, kernel_size=8, stride=2, padding=3)  # k8
        self.enc_conv5 = nn.Conv2d(64, 96, kernel_size=7, stride=1, padding=3)  # k7
        self.enc_conv6 = nn.Conv2d(96, 96, kernel_size=8, stride=2, padding=3)  # k8
        self.enc_conv7 = nn.Conv2d(96, 128, kernel_size=7, stride=1, padding=3)  # k7
        
        # Bottleneck layers
        self.bottleneck_conv1 = nn.Conv2d(128, 128, kernel_size=8, stride=2, padding=3)  # k8
        self.bottleneck_conv2 = nn.Conv2d(128, 160, kernel_size=7, stride=1, padding=3)  # k7
        
        # Decoding layers (Upsampling)
        self.dec_conv1 = nn.ConvTranspose2d(160 + 128, 160, kernel_size=8, stride=2, padding=3, output_padding=1)  # k8
        self.dec_conv2 = nn.Conv2d(160, 128, kernel_size=8, stride=1, padding=3)  # k7
        self.dec_conv3 = nn.ConvTranspose2d(128+128, 128, kernel_size=8, stride=2, padding=3, output_padding=1)  # k8
        self.dec_conv4 = nn.Conv2d(128, 96, kernel_size=8, stride=1, padding=3)  # k7
        self.dec_conv5 = nn.ConvTranspose2d(96+96, 96, kernel_size=8, stride=2, padding=2, output_padding=1)  # k6
        self.dec_conv6 = nn.Conv2d(96, 64, kernel_size=8, stride=1, padding=2)  # k5
        self.dec_conv7 = nn.ConvTranspose2d(64+64, 64, kernel_size=7, stride=2, padding=3, output_padding=1)  # k8
        self.dec_conv8 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)  # k5
        self.dec_conv9 = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)  # k5

        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Encoding path
        e1 = self.relu(self.enc_conv1(x))
        if DEBUG:
            print("e1: ", e1.shape)
        e2 = self.relu(self.enc_conv2(e1))
        if DEBUG:
            print("e2: ", e2.shape)
        e3 = self.relu(self.enc_conv3(e2))
        if DEBUG:
            print("e3: ", e3.shape)
        e4 = self.relu(self.enc_conv4(e3))
        if DEBUG:
            print("e4: ", e4.shape)
        e5 = self.relu(self.enc_conv5(e4))
        if DEBUG:
            print("e5: ", e5.shape)
        e6 = self.relu(self.enc_conv6(e5))
        if DEBUG:
            print("e6: ", e6.shape)
        e7 = self.relu(self.enc_conv7(e6))
        if DEBUG:
            print("e7: ", e7.shape)

        # Bottleneck
        if DEBUG:
            print(e7.shape)
        b1 = self.relu(self.bottleneck_conv1(e7))
        if DEBUG:
            print("b1: ", b1.shape)
        b2 = self.relu(self.bottleneck_conv2(b1))
        if DEBUG:
            print("b2: ", b2.shape)
        # Decoding path with skip connections
        
        c1 = torch.cat((b1, b2), dim=1)  # Concatenate with encoding layer e6
        if DEBUG:
            print("c1: ", b2.shape)
        d1 = self.relu(self.dec_conv1(c1))
        if DEBUG:
            print("d1: ", d1.shape)
        d2 = self.relu(self.dec_conv2(d1))
        if DEBUG:
            print("d2: ", d2.shape)

        c2 = torch.cat((e7, d2), dim=1)  # Concatenate with encoding layer e4
        if DEBUG:
            print("c2: ", c2.shape)
        d3 = self.relu(self.dec_conv3(c2))
        if DEBUG:
            print("d3: ", d3.shape)
        d4 = self.relu(self.dec_conv4(d3))
        if DEBUG:
            print("d4: ", d4.shape)

        c3 = torch.cat((d4, e5), dim=1)  # Concatenate with encoding layer e2
        if DEBUG:
            print("c3: ", c3.shape)
        d5 = self.relu(self.dec_conv5(c3))
        if DEBUG:
            print("d5: ", d5.shape)
        d6 = self.relu(self.dec_conv6(d5))
        if DEBUG:
            print("d6: ", d6.shape)

        c4 = torch.cat((d6, e3), dim=1)  # Concatenate with encoding layer e2
        if DEBUG:
            print("c4: ", c4.shape)
        d7 = self.relu(self.dec_conv7(c4))
        if DEBUG:
            print("d7: ", d7.shape)
        d8 = self.relu(self.dec_conv8(d7))
        if DEBUG:
            print("d8: ", d8.shape)
        out = self.relu(self.dec_conv9(d8))
        # Final output layer
        #out = self.dec_conv7(d9)
        
        return out
