"""
Event Detection Model for SoccerNet
ResNet18 + LSTM architecture for temporal event detection
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EventDetectionModel(nn.Module):
    """
    Event detection model combining CNN (ResNet18) and LSTM for temporal modeling.
    
    Architecture:
    1. ResNet18 backbone (pretrained on ImageNet) - per-frame feature extraction
    2. LSTM - temporal modeling across frame sequence
    3. Fully connected layers - classification
    """
    
    def __init__(
        self,
        num_classes=17,
        sequence_length=16,
        pretrained=True,
        hidden_dim=512,
        lstm_layers=2,
        dropout=0.5
    ):
        """
        Args:
            num_classes: Number of event classes
            sequence_length: Number of frames in input sequence
            pretrained: Whether to use pretrained ResNet18
            hidden_dim: Hidden dimension for LSTM
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(EventDetectionModel, self).__init__()
        
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # ========== CNN Backbone (ResNet18) ==========
        if pretrained:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
                
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        except AttributeError:
            resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer
        # ResNet18 output: [batch, 512, 7, 7] after avgpool
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension after ResNet18
        self.feature_dim = 512
        
        # ========== LSTM for Temporal Modeling ==========
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # ========== Classification Head ==========
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of LSTM and classifier."""
        for module in [self.lstm, self.classifier]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    if 'lstm' in str(module):
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.kaiming_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, sequence_length, channels, height, width]
               Shape: [B, T, C, H, W] where T=sequence_length
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size, sequence_length, C, H, W = x.shape
        
        # ========== Extract features from each frame ==========
        # Reshape to process all frames at once
        x = x.view(batch_size * sequence_length, C, H, W)
        
        # Extract features using ResNet18
        features = self.feature_extractor(x)  # [B*T, 512, 1, 1]
        features = features.view(batch_size * sequence_length, -1)  # [B*T, 512]
        
        # Reshape back to sequence
        features = features.view(batch_size, sequence_length, -1)  # [B, T, 512]
        
        # ========== Temporal modeling with LSTM ==========
        # LSTM output: (output, (h_n, c_n))
        # output: [B, T, hidden_dim]
        # h_n: [num_layers, B, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use the last hidden state for classification
        # h_n[-1]: last layer's hidden state [B, hidden_dim]
        last_hidden = h_n[-1]  # [B, hidden_dim]
        
        # Alternative: Use the last time step output
        # last_hidden = lstm_out[:, -1, :]  # [B, hidden_dim]
        
        # ========== Classification ==========
        logits = self.classifier(last_hidden)  # [B, num_classes]
        
        return logits
    
    def extract_features(self, x):
        """
        Extract features without classification (for analysis).
        
        Args:
            x: Input tensor [B, T, C, H, W]
        
        Returns:
            features: LSTM features [B, hidden_dim]
        """
        batch_size, sequence_length, C, H, W = x.shape
        
        # Extract CNN features
        x = x.view(batch_size * sequence_length, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(batch_size * sequence_length, -1)
        features = features.view(batch_size, sequence_length, -1)
        
        # Get LSTM features
        lstm_out, (h_n, c_n) = self.lstm(features)
        last_hidden = h_n[-1]
        
        return last_hidden


class EventDetectionModelV2(nn.Module):
    """
    Alternative architecture using 3D CNN for spatiotemporal feature extraction.
    Can be used as a lighter alternative to ResNet+LSTM.
    """
    
    def __init__(self, num_classes=17, dropout=0.5):
        super(EventDetectionModelV2, self).__init__()
        
        self.features = nn.Sequential(
            # Conv3D: (in_channels, out_channels, kernel_size)
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Input: [B, T, C, H, W]
        # Conv3D expects: [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.features(x)
        x = self.classifier(x)
        
        return x


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test code
if __name__ == "__main__":
    # Test model
    batch_size = 4
    sequence_length = 16
    num_classes = 17
    
    # Create dummy input
    x = torch.randn(batch_size, sequence_length, 3, 224, 224)
    
    print("Testing EventDetectionModel (ResNet18 + LSTM):")
    model = EventDetectionModel(
        num_classes=num_classes,
        sequence_length=sequence_length,
        pretrained=False  # Set to False for testing without downloading
    )
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {num_classes})")
    
    assert output.shape == (batch_size, num_classes), "Output shape mismatch!"
    print("✓ Model test passed!")
    
    print("\n" + "="*70)
    print("Testing EventDetectionModelV2 (3D CNN):")
    model_v2 = EventDetectionModelV2(num_classes=num_classes)
    print(f"Total parameters: {count_parameters(model_v2):,}")
    
    output_v2 = model_v2(x)
    print(f"Output shape: {output_v2.shape}")
    assert output_v2.shape == (batch_size, num_classes), "Output shape mismatch!"
    print("✓ Model V2 test passed!")
