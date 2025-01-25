import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_p=0.0, bn = False, activation='relu'):
        super().__init__()

        # Create a list of layer sizes
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create a list of linear layers using ModuleList
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes)-1)
        ])
        for layer in self.layers:
            #nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.weight, 0.1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.01)
                

        # Create a list of batch normalization layers using ModuleList
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(layer_sizes[i+1])
            for i in range(len(hidden_sizes))
        ])

        # Create a dropout layer
        self.dropout = nn.Dropout(dropout_p)
        self.bn = bn
        self.activation = activation

    def forward(self, x):
        # Iterate over the linear layers and apply them sequentially to the input
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.bn:
                x = self.batch_norms[i](x)
            activation_fn = getattr(nn.functional, self.activation)
            x = activation_fn(x)
            x = self.dropout(x)
        # Apply the final linear layer to get the output
        x = self.layers[-1](x)
        return x


class Basic_ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        '''
        ResNet18 model with the final fully connected layer replaced to match the number of classes.

        '''
        super(ResNet18, self).__init__()
        
        # Load the pre-trained ResNet18 model
        self.model = models.resnet18(pretrained=True)

        # Replace the final fully connected layer to match the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Initialize the weights of the final layer with constant values
        nn.init.constant_(self.model.fc.weight, 0.1)
        if self.model.fc.bias is not None:
            nn.init.constant_(self.model.fc.bias, 0.1)

        # Freeze all layers except the last one
        for param in self.model.parameters():
            param.requires_grad = False

        # Ensure the final layer is trainable
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
class ResNet18_old(nn.Module):
    def __init__(self, input_channels, hidden_sizes, num_classes, dropout_p=0.0, bn=False, activation='relu'):
        super(ResNet18, self).__init__()

        # Base ResNet-18 backbone
        self.resnet = torchvision.models.resnet18(pretrained=False)

        # Modify the first convolution layer to accept specified input channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Extract feature extractor layers from ResNet
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        # Define fully connected layers with specified hidden sizes
        layer_sizes = [self.resnet.fc.in_features] + hidden_sizes + [num_classes]
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)
        ])

        # Initialize weights and biases for fully connected layers
        for layer in self.layers:
            nn.init.constant_(layer.weight, 0.1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.01)

        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(layer_sizes[i+1]) for i in range(len(hidden_sizes))
        ])

        self.dropout = nn.Dropout(dropout_p)
        self.bn = bn
        self.activation = activation

    def forward(self, x):
        # Pass through the ResNet feature extractor
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps

        # Pass through fully connected layers
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.bn:
                x = self.batch_norms[i](x)
            activation_fn = getattr(nn.functional, self.activation)
            x = activation_fn(x)
            x = self.dropout(x)

        # Apply the final layer
        x = self.layers[-1](x)
        return x
#-----------------------------------------------------------------------------------------------------------------

class ResNet32_34(nn.Module):
    def __init__(self, input_channels, hidden_sizes, num_classes, dropout_p=0.0, bn=False, activation='relu'):
        super(ResNet32, self).__init__()

        # Base ResNet-34 backbone (approximating ResNet-32)
        self.resnet = torchvision.models.resnet34(pretrained=False)

        # Modify the first convolution layer to accept specified input channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Extract feature extractor layers from ResNet
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        # Define fully connected layers with specified hidden sizes
        layer_sizes = [self.resnet.fc.in_features] + hidden_sizes + [num_classes]
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)
        ])

        # Initialize weights and biases for fully connected layers
        for layer in self.layers:
            nn.init.constant_(layer.weight, 0.1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.01)

        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(layer_sizes[i+1]) for i in range(len(hidden_sizes))
        ])

        self.dropout = nn.Dropout(dropout_p)
        self.bn = bn
        self.activation = activation

    def forward(self, x):
        # Pass through the ResNet feature extractor
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps

        # Pass through fully connected layers
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.bn:
                x = self.batch_norms[i](x)
            activation_fn = getattr(nn.functional, self.activation)
            x = activation_fn(x)
            x = self.dropout(x)

        # Apply the final layer
        x = self.layers[-1](x)
        return x
#-------------------------------------------------------------------------------------------------------------------------------------------------



class BasicBlock(nn.Module):
    expansion = 1  # BasicBlock does not expand channels

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


class ResNet32(nn.Module):
    def __init__(self, num_classes=20):
        super(ResNet32, self).__init__()
        self.in_channels = 16  # Start with 16 channels for CIFAR

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Define layers
        self.layer1 = self._make_layer(16, 5, stride=1)  # 5 blocks
        self.layer2 = self._make_layer(32, 5, stride=2)  # 5 blocks, downsample
        self.layer3 = self._make_layer(64, 5, stride=2)  # 5 blocks, downsample

        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)

        # Apply weight initialization
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride):
        layers = []

        # Downsampling layer if needed
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # First block with downsampling
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Global Average Pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_channels = 64  # Start with 64 channels

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()

        # Define layers
        self.layer1 = self._make_layer(64, 2, stride=1)  # 2 blocks
        self.layer2 = self._make_layer(128, 2, stride=2)  # 2 blocks, downsample
        self.layer3 = self._make_layer(256, 2, stride=2)  # 2 blocks, downsample
        self.layer4 = self._make_layer(512, 2, stride=2)  # 2 blocks, downsample

        # Fully connected layer
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride):
        layers = []

        # Downsampling layer if needed
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # First block with downsampling
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)