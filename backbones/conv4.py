import torch.nn as nn

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Conv4(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, hid_channels=64):
        super(Conv4, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hid_channels),
            conv3x3(hid_channels, hid_channels),
            conv3x3(hid_channels, hid_channels),
            conv3x3(hid_channels, out_channels)
        )

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        return outputs


def conv4(**kwargs):
    return Conv4(**kwargs)