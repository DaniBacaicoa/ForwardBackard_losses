import torch
import torch.nn as nn

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
    
class ResNet18(nn.Module):
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
-----------------------------------------------------------------------------------------------------------------
class ResNet32(nn.Module):
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
