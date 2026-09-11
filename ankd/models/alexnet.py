"""AlexNet / AlexNet-Half for 32x32 CIFAR images.

From the ZSKD supplementary, Table 2 (Nayak et al., "Zero-Shot Knowledge
Distillation in Deep Networks", ICML 2019), cross-checked against the authors'
released model_alex_full.py / model_alex_half.py:

    AlexNet-Half is derived from AlexNet by taking half of the convolutional
    filters and half of the neurons in the fully connected layers, except in the
    classification layer.

Two notes on the TensorFlow -> PyTorch port:

* LRN. TF's local_response_normalization(depth_radius=2, alpha=1e-4, beta=0.75,
  bias=1.0) sums over a 5-channel window and does NOT divide alpha by the window
  size, while torch.nn.LocalResponseNorm does. Passing alpha*5 makes the two
  numerically identical.
* Init. The reference initialises every weight from N(0, 0.01); that is available
  as paper_init=True, but the default is Kaiming, which trains far more reliably
  without TF-era learning-rate tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetCIFAR(nn.Module):
    """Block order follows the reference implementation:

        conv -> ReLU -> LRN -> [maxpool] -> BN
        fc   -> ReLU -> dropout -> BN
    """

    def __init__(self, widths, num_classes=10, dropout=0.5, paper_init=False):
        super().__init__()
        c1, c2, c3, c4, c5, f1, f2 = widths

        def lrn():
            return nn.LocalResponseNorm(size=5, alpha=1e-4 * 5, beta=0.75, k=1.0)

        # 'SAME' padding at stride 1: k=5 -> pad 2, k=3 -> pad 1.
        # Pooling is 3x3 / stride 2 'VALID': 32 -> 15 -> 7 -> 3.
        self.conv1, self.lrn1 = nn.Conv2d(3, c1, 5, 1, 2), lrn()
        self.pool1, self.bn1 = nn.MaxPool2d(3, 2), nn.BatchNorm2d(c1)

        self.conv2, self.lrn2 = nn.Conv2d(c1, c2, 5, 1, 2), lrn()
        self.pool2, self.bn2 = nn.MaxPool2d(3, 2), nn.BatchNorm2d(c2)

        self.conv3, self.bn3 = nn.Conv2d(c2, c3, 3, 1, 1), nn.BatchNorm2d(c3)
        self.conv4, self.bn4 = nn.Conv2d(c3, c4, 3, 1, 1), nn.BatchNorm2d(c4)

        self.conv5, self.pool5 = nn.Conv2d(c4, c5, 3, 1, 1), nn.MaxPool2d(3, 2)
        self.bn5 = nn.BatchNorm2d(c5)

        self.fc1, self.drop1 = nn.Linear(3 * 3 * c5, f1), nn.Dropout(dropout)
        self.bn6 = nn.BatchNorm1d(f1)
        self.fc2, self.drop2 = nn.Linear(f1, f2), nn.Dropout(dropout)
        self.bn7 = nn.BatchNorm1d(f2)
        self.fc3 = nn.Linear(f2, num_classes)

        self._init_weights(paper_init)

    def _init_weights(self, paper_init):
        if paper_init:
            ones = {self.conv2, self.conv4, self.conv5}   # original AlexNet convention
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, 1.0 if m in ones else 0.0)
            return
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.bn1(self.pool1(self.lrn1(F.relu(self.conv1(x)))))
        x = self.bn2(self.pool2(self.lrn2(F.relu(self.conv2(x)))))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(self.pool5(F.relu(self.conv5(x))))
        x = torch.flatten(x, 1)
        x = self.bn6(self.drop1(F.relu(self.fc1(x))))
        x = self.bn7(self.drop2(F.relu(self.fc2(x))))
        return self.fc3(x)


def alexnet(num_classes=10, **kw):
    return AlexNetCIFAR((48, 128, 192, 192, 128, 512, 256), num_classes, **kw)


def alexnet_half(num_classes=10, **kw):
    return AlexNetCIFAR((24, 64, 96, 96, 64, 256, 128), num_classes, **kw)
