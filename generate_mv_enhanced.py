import os
import torch
torch.cuda.set_device(0)
from PIL import Image

# Set CUDA environment before any imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Performance optimizations - MUST be before other imports
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

try:
    torch.backends.cuda.enable_flash_sdp(True)
    print("✓ Flash attention enabled")
except:
    print("⚠ Flash attention not available")

print("🚀 Loading model...")

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

print("🚀 loaded model...")

def generate_3d_mesh(image_paths, output_path="output.glb", model_version="2.1", device='cuda', dtype=torch.float16, use_safetensors=False, num_inference_steps=20, octree_resolution=380, num_chunks=20000, generator=torch.manual_seed(12345), output_type='trimesh'):
    """
    Generate 3D mesh from multi-view images
    
    Args:
        image_paths (dict): Dictionary with image paths for different views
        output_path (str): Output file path
        model_version (str): "2mv" for multi-view model or "2.1" for latest model
    """
    
    ckpt_path = "./models/hunyuan3d-dit-v2-1/hunyuan3d-dit-v2-1/model.fp16.ckpt"
    config_path = "./models/hunyuan3d-dit-v2-1/hunyuan3d-dit-v2-1/config.yaml"
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.set_default_tensor_type('torch.cuda.HalfTensor')
    # Load model - VRAM only
    print(f"🚀 Loading model to VRAM...")
    with torch.cuda.device(0):
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path=ckpt_path,
            config_path=config_path,
            device='cuda',
            dtype=torch.float16,
            use_safetensors=False
        )
    
    # Initialize background remover
    rembg = BackgroundRemover()
    
    # Process images
    print("🖼️ Processing images...")
    images = {}
    
    # Handle both single image and multi-view inputs
    if isinstance(image_paths, dict):
        # Multi-view input
        for key, path in image_paths.items():
            img = Image.open(path).convert("RGBA")
            if img.mode == 'RGB':
                img = rembg(img)
            images[key] = img
    else:
        # Single image input
        img = Image.open(image_paths).convert("RGBA")
        if img.mode == 'RGB':
            img = rembg(img)
        images = img
    
    # Generate mesh
    print("🎯 Generating mesh...")
    with torch.inference_mode():
        mesh = pipeline(
            image=images,
            num_inference_steps=num_inference_steps,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            generator=generator,
            output_type=output_type
        )[0]
    
    # Export
    mesh.export(output_path)
    print(f"💾 Mesh saved as {output_path}")
    return output_path

def generate_multiview_mesh(image_paths, output_path="mv_output.glb"):
    """Generate mesh using multi-view optimized model"""
    return generate_3d_mesh(image_paths, output_path, model_version="2mv")

def generate_latest_mesh(image_paths, output_path="latest_output.glb"):
    """Generate mesh using latest Hunyuan3D-2.1 model"""
    return generate_3d_mesh(image_paths, output_path, model_version="2.1")