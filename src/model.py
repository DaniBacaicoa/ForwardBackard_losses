import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

'''
class Basic_ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        
        ResNet18 model with the final fully connected layer replaced to match the number of classes.

        
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
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        #if self.downsample is not None:
        #    identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

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


# Define ResNet Architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
        '''

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out

class ResNet18CIFAR(nn.Module):
    """A ResNet-18 adapted for CIFAR-10: 
       - 4 layers with [2, 2, 2, 2] BasicBlocks
       - 3×3 conv (stride=1) at the stem
    """
    def __init__(self, block, layers, num_classes=10):
        super(ResNet18CIFAR, self).__init__()
        self.in_channels = 64  # Start with 64 channels (as in standard ResNet-18)
        
        # Stem: for CIFAR-10, we can do a simple 3×3 conv, stride=1, no pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 4 layers (stages), typical ResNet-18 config is [2, 2, 2, 2]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialization (Kaiming)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        # If we're changing the spatial dimension (stride != 1) or 
        # the number of channels, we need a downsample
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,
                          out_channels * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block in this layer
        layers.append(block(self.in_channels, out_channels, 
                            stride=stride, downsample=downsample))
        self.in_channels = out_channels * block.expansion

        # Remaining blocks in this layer
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # (N, 64, 32, 32)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # (N, 64, 32, 32)
        x = self.layer2(x)  # (N,128, 16, 16)
        x = self.layer3(x)  # (N,256,  8,  8)
        x = self.layer4(x)  # (N,512,  4,  4)

        x = self.avgpool(x) # (N,512,1,1)
        x = torch.flatten(x, 1)  # (N,512)
        x = self.fc(x)          # (N,10)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 16
        
        # Initial (stem) convolution: 3 -> 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)
        
        # Layers (stages)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        
        # Global average pool + fully-connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(64 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # If we are changing spatial resolution (stride != 1) 
        # or changing channel dimension, we need a downsample
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,
                          out_channels * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block in the layer
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks in the layer
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # e.g., keeps resolution if stride=1
        x = self.layer2(x)  # e.g., halves resolution if stride=2
        x = self.layer3(x)  # e.g., halves resolution again
        
        x = self.avgpool(x)       # Global average pooling
        x = torch.flatten(x, 1)   # Flatten to (batch_size, channels)
        x = self.fc(x)            # Fully connected layer

        return x


class ResNet_18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_18, self).__init__()
        
        # Load a base ResNet-18 (no pretrained weights)
        # If your PyTorch version uses 'weights' argument, set weights=None
        # If it uses 'pretrained' argument, set pretrained=False
        self.resnet = models.resnet18(weights=None)
        
        # 1) Modify the first convolution layer:
        #    7×7 kernel, stride=2 --> 3×3 kernel, stride=1
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # 2) Remove the max-pool layer
        self.resnet.maxpool = nn.Identity()
        
        # 3) Replace the final FC layer to match CIFAR-10 classes
        self.resnet.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        return self.resnet(x)
class ResNet50(nn.Module):
    """
    A ResNet-50 model pre-trained on ImageNet, adapted for the Clothing1M dataset.

    The final fully connected layer is replaced to match the number of classes
    in the Clothing1M dataset (14 classes).
    """
    def __init__(self, num_classes=14, fine_tune_all=False):
        """
        Initializes the ResNet50Clothing1M model.

        Args:
            num_classes (int): The number of output classes (default: 14 for Clothing1M).
            pretrained (bool): Whether to load weights pre-trained on ImageNet (default: True).
        """
        super(ResNet50, self).__init__()

        self.fine_tune_all = fine_tune_all

        # Load the pre-trained ResNet-50 model
        # Use weights=models.ResNet50_Weights.IMAGENET1K_V1 for older torchvision
        # or weights=models.ResNet50_Weights.DEFAULT for newer versions
        weights = models.ResNet50_Weights.DEFAULT

        self.resnet50 = models.resnet50(weights=weights)

        # Get the number of input features for the original fully connected layer
        num_ftrs = self.resnet50.fc.in_features

        # Replace the final fully connected layer (fc) with a new one
        # The new layer has the same number of input features but outputs
        # `num_classes` features, suitable for the Clothing1M dataset.
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
        if not fine_tune_all:
            print("Freezing base model parameters. Only the final classifier will be trained.")
            # Freeze all parameters first
            for param in self.resnet50.parameters():
                param.requires_grad = False
            # Unfreeze the parameters of the final layer (fc)
            for param in self.resnet50.fc.parameters():
                param.requires_grad = True
        else:
            print("All model parameters will be fine-tuned.")


    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor (batch of images).

        Returns:
            torch.Tensor: The output tensor (logits for each class).
        """
        return self.resnet50(x)