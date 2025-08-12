import os
import torch
from PIL import Image

# Performance optimizations - MUST be before other imports
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

try:
    torch.backends.cuda.enable_flash_sdp(True)
    print("✓ Flash attention enabled")
except:
    print("⚠ Flash attention not available")

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

def generate_3d_mesh(image_paths, output_path="output.glb", model_version="2mv"):
    """
    Generate 3D mesh from multi-view images
    
    Args:
        image_paths (dict): Dictionary with image paths for different views
        output_path (str): Output file path
        model_version (str): "2mv" for multi-view model or "2.1" for latest model
    """
    
    # Set model path
    os.environ['HY3DGEN_MODELS'] = '/workspace/pro/Hunyuan3D-2/model-kp/hunyuan3d-download'
    
    # Select model based on version
    if model_version == "2mv":
        model_name = 'hunyuan3d-dit-v2-mv'
        print("🔧 Using Hunyuan3D-2mv (Multi-View optimized)")
    elif model_version == "2.1":
        model_name = 'hunyuan3d-dit-v2-1'
        print("🔧 Using Hunyuan3D-2.1 (Latest)")
    else:
        raise ValueError("model_version must be '2mv' or '2.1'")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Load model
    print(f"🚀 Loading {model_name}...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_name,
        use_safetensors=True,
        device='cuda',
        torch_dtype=torch.float16
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
            num_inference_steps=20,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type='trimesh'
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

# if __name__ == "__main__":
#     # Multi-view image paths
#     mv_image_paths = {
#         "front": 'assets/example_mv_images/1/front.png',
#         "back": 'assets/example_mv_images/1/back.png',
#         "left": 'assets/example_mv_images/1/left.png'
#     }
    
#     # Single image path (for 2.1 model)
#     single_image_path = 'assets/example_mv_images/1/front.png'
    
#     print("🔄 Testing Multi-View model (2mv)...")
#     generate_multiview_mesh(mv_image_paths, "demo_mv_output.glb")
    
#     print("\n🔄 Testing Latest model (2.1) with single image...")
#     generate_latest_mesh(single_image_path, "demo_latest_output.glb")
    
#     print("\n🔄 Testing Latest model (2.1) with multi-view...")
#     generate_latest_mesh(mv_image_paths, "demo_latest_mv_output.glb")