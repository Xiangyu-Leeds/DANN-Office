import torch.nn as nn
from functions import ReverseLayerF
import torchvision.models as models


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = models.alexnet(pretrained=True).features
        alexnet_output_dim = 256 * 6 * 6
        # self.feature = nn.Sequential()
        # self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        # self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        # self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu1', nn.ReLU(True))
        # self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        # self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        # self.feature.add_module('f_drop1', nn.Dropout2d())
        # self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        # self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(alexnet_output_dim, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 31))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(alexnet_output_dim, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha, batch_size):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
        feature = self.feature(input_data)
        feature = feature.view(batch_size, -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output
