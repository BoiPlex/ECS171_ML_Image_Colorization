"""
Flask Backend for Image Colorization
"""

from flask import Flask, send_from_directory, request, jsonify
from PIL import Image
import torch
import numpy as np
import io
import os

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import model from backend package
from model import ColorizationCNN, load_model, preprocess_image, compute_orthonormal_basis, reconstruct_color, LUMINANCE_WEIGHTS, MODEL_HEIGHT, MODEL_WIDTH

# Get the path to the frontend folder
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')

app = Flask(__name__, static_folder=frontend_path, static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Load model on startup
checkpoint_path = os.path.join(os.path.dirname(__file__), 'cifar10_colorizer.pth')
model = load_model(ColorizationCNN, checkpoint_path, device)
print(f"ColorizationCNN loaded from {checkpoint_path} on {device}")


# Compute color basis on startup
w = LUMINANCE_WEIGHTS.to(device)
u1, u2 = compute_orthonormal_basis(w, device)
print(f"Color basis computed: w={w.tolist()}, u1={u1.tolist()}, u2={u2.tolist()}")


def postprocess_output(L_tensor, alpha_beta_tensors, image_size=(MODEL_HEIGHT, MODEL_WIDTH)):
    """
    Reconstruct RGB image from luminance and alpha/beta coefficients.
    
    The model outputs (alpha, beta) coefficients in an orthonormal color basis.
    We reconstruct the full RGB image using:
        V = (Y / (w·w)) * w + alpha * u1 + beta * u2
    
    Args:
        L_tensor: Input luminance (1, 1, H, W)
        alpha_beta_tensors: Tuple of (alpha, beta) from model, each (1, 1, H, W)
        image_size: (H, W) tuple for reshaping
        
    Returns:
        PIL.Image: Colorized RGB image
    """
    alpha_pred, beta_pred = alpha_beta_tensors
    
    # Extract and move to CPU
    L = L_tensor.detach().cpu().squeeze(0).squeeze(0)  # (H, W)
    alpha = alpha_pred.detach().cpu().squeeze(0).squeeze(0)  # (H, W)
    beta = beta_pred.detach().cpu().squeeze(0).squeeze(0)  # (H, W)
    
    # Flatten for reconstruction
    L_flat = L.reshape(-1)  # (H*W,)
    alpha_flat = alpha.reshape(-1)  # (H*W,)
    beta_flat = beta.reshape(-1)  # (H*W,)
    
    # Reconstruct colors using orthonormal basis
    V = reconstruct_color(L_flat, alpha_flat, beta_flat, w, u1, u2)  # (H*W, 3)
    
    # Clamp to valid RGB range and reshape
    V = V.clamp(0, 1)  # (H*W, 3)
    rgb_array = V.reshape(image_size[0], image_size[1], 3).numpy() * 255  # (H, W, 3)
    rgb_array = rgb_array.astype(np.uint8)
    
    # Convert to PIL Image
    image_pil = Image.fromarray(rgb_array, mode='RGB')
    return image_pil


@app.route('/')
def index():
    """Serve home page"""
    return send_from_directory(frontend_path, 'project.html')


@app.route('/<path:filename>')
def serve_file(filename):
    """Serve static files from frontend folder"""
    return send_from_directory(frontend_path, filename)


@app.route('/colorize', methods=['POST'])
def colorize():
    """
    Colorize an image.
    
    Accepts: image file (any size, any color format)
    Returns: colorized RGB image
    """
    # Check if image file is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Open image
    image = Image.open(io.BytesIO(file.read()))
    
    # Preprocess image
    tensor_input = preprocess_image(image).to(device)
    
    # Run through model
    with torch.no_grad():
        alpha_beta_output = model(tensor_input)
    
    # Postprocess output (reconstruct RGB from alpha/beta and luminance)
    colorized_image = postprocess_output(tensor_input, alpha_beta_output, (MODEL_HEIGHT, MODEL_WIDTH))
    
    # Save to bytes and return
    img_bytes = io.BytesIO()
    colorized_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue(), 200, {'Content-Type': 'image/png'}


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': True
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
