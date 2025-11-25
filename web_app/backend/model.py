"""
Image Colorization Model

Contains ColorizationCNN architecture, color basis utilities, and model loading.
Based on orthonormal color basis decomposition: V = (Y/w·w)*w + alpha*u1 + beta*u2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Model input size constants
# ColorizationCNN expects square images with dimensions divisible by 8
# (due to 3 stride-2 convolutions followed by 3 stride-2 transposed convolutions)
MODEL_WIDTH = 128
MODEL_HEIGHT = 128

# Luminance weights for converting RGB to grayscale
# Standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
LUMINANCE_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])


def compute_orthonormal_basis(w, device='cpu'):
    """Compute orthonormal basis (u1, u2) perpendicular to luminance vector w.
    
    Args:
        w: (3,) luminance weights vector
        device: torch device
    
    Returns:
        u1, u2: orthonormal basis vectors perpendicular to w
    """
    w = w.to(device)
    w_norm = w / torch.norm(w)
    tmp = torch.tensor([1., 0., 0.], device=device) if abs(w_norm[0]) < 0.9 else torch.tensor([0., 1., 0.], device=device)
    u1 = tmp - w_norm * torch.dot(tmp, w_norm)
    u1 = u1 / (torch.norm(u1) + 1e-8)
    u2 = torch.linalg.cross(w_norm, u1)
    u2 = u2 / (torch.norm(u2) + 1e-8)
    return u1, u2


def reconstruct_color(Y_target, alpha, beta, w, u1, u2):
    """Reconstruct RGB color from luminance and coefficients in orthonormal basis.
    
    V = (Y / (w·w)) * w + alpha * u1 + beta * u2
    
    Args:
        Y_target: (N,) luminance values
        alpha: (N,) coefficient for u1 basis vector
        beta: (N,) coefficient for u2 basis vector
        w: (3,) luminance weight vector
        u1: (3,) first orthonormal basis vector
        u2: (3,) second orthonormal basis vector
    
    Returns:
        V: (N, 3) RGB colors
    """
    w2 = torch.dot(w, w)
    v_parallel = (Y_target / w2).unsqueeze(1) * w
    v_perp = alpha.unsqueeze(1) * u1 + beta.unsqueeze(1) * u2
    return v_parallel + v_perp


class ColorizationCNN(nn.Module):
    """Predicts alpha and beta (coefficients in orthonormal color basis)"""
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)  # 32 x H/2 x W/2
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 64 x H/4 x W/4
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1) # 128 x H/8 x W/8
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.tconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.tbn1 = nn.BatchNorm2d(64)

        self.tconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.tbn2 = nn.BatchNorm2d(32)

        self.tconv3 = nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1)

    def forward(self, L):
        x = F.relu(self.bn1(self.conv1(L)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.tbn1(self.tconv1(x)))
        x = F.relu(self.tbn2(self.tconv2(x)))
        x = self.tconv3(x)   # final alpha, beta
        a = x[:, 0:1]
        b = x[:, 1:2]
        return a, b


def load_model(model_class, checkpoint_path: str, device):
    """Load model with weights from checkpoint.
    
    Args:
        model_class: The model class to instantiate (e.g., ColorizationCNN)
        checkpoint_path: Path to the checkpoint .pth file
        device: torch device (cpu or cuda)
        
    Returns:
        Model loaded with weights and set to eval mode on the specified device
    """
    model = model_class()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_pil):
    """Preprocess image for colorization model.
    
    1. Resize to model input size (MODEL_WIDTH x MODEL_HEIGHT)
    2. Convert to RGB (if not already)
    3. Convert to tensor
    4. Compute luminance Y = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image_pil: PIL Image (RGB or RGBA)
        
    Returns:
        torch.Tensor: Luminance image (1, 1, H, W), values in [0, 1]
    """
    # Ensure RGB
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    # Resize to model input size
    image_resized = image_pil.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.LANCZOS)
    
    # Convert to numpy array (H, W, 3) with values [0, 255]
    image_array = np.array(image_resized, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    
    # Compute luminance: Y = 0.299*R + 0.587*G + 0.114*B
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # (3,)
    luminance = np.dot(image_array, w)  # (H, W)
    
    # Convert to tensor and add batch/channel dimensions
    tensor = torch.from_numpy(luminance).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    return tensor
