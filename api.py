from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import tempfile
import os
import shutil
import torch
# Set CUDA environment before any imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)
from PIL import Image

# Import the pipeline and background remover
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


# Global variables to store pre-loaded models
pipeline_cache = {}
rembg = None

def load_models_to_gpu():
    """Load all models to GPU at startup"""
    global pipeline_cache, rembg
    
    print("🚀 Pre-loading models to GPU...")
    
    # Clear GPU cache first
    torch.cuda.empty_cache()
    
    # Model configurations
    model_configs = {
        "2.1": {
            "ckpt": "./models/hunyuan3d-dit-v2-1/hunyuan3d-dit-v2-1/model.fp16.ckpt",
            "config": "./models/hunyuan3d-dit-v2-1/hunyuan3d-dit-v2-1/config.yaml"
        },
        "2mv": {
            "ckpt": "./models/hunyuan3d-dit-v2-mv/hunyuan3d-dit-v2-mv/model.fp16.ckpt",
            "config": "./models/hunyuan3d-dit-v2-mv/hunyuan3d-dit-v2-mv/config.yaml"
        }
    }
    
    # Load each model to GPU
    for version, config in model_configs.items():
        if os.path.exists(config["ckpt"]) and os.path.exists(config["config"]):
            print(f"📦 Loading Hunyuan3D-{version} to GPU...")
            torch.set_default_tensor_type('torch.cuda.HalfTensor')
            with torch.cuda.device(0):
                pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                    ckpt_path=config["ckpt"],
                    config_path=config["config"],
                    device='cuda',
                    dtype=torch.float16,
                    use_safetensors=False
                )
            
            # Enable CPU offloading for memory efficiency
            #pipeline.enable_model_cpu_offload()
            
            pipeline_cache[version] = pipeline
            print(f"✅ Hunyuan3D-{version} loaded to GPU")
        else:
            print(f"⚠️ Hunyuan3D-{version} model files not found, skipping...")
    
    # Load background remover
    print("📦 Loading background remover...")
    rembg = BackgroundRemover()
    print("✅ Background remover loaded")
    
    # Report GPU memory usage
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1e9
        print(f"🎯 Total GPU memory used: {memory_gb:.2f} GB")
    
    print("🎉 All models pre-loaded successfully!")

def generate_3d_mesh_cached(image_paths, output_path="output.glb", model_version="2.1"):
    """Generate 3D mesh using pre-loaded models"""
    global pipeline_cache, rembg
    
    # Get the cached pipeline
    if model_version not in pipeline_cache:
        raise ValueError(f"Model {model_version} not loaded. Available models: {list(pipeline_cache.keys())}")
    
    pipeline = pipeline_cache[model_version]
    
    print(f"🔧 Using pre-loaded Hunyuan3D-{model_version}")
    
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

# Lifespan context manager for startup/shutdown
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup: Load models when the server starts
    load_models_to_gpu()
    yield
    # Shutdown: Cleanup if needed
    print("🛑 Server shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(title="Hunyuan3D Multi-Model API", lifespan=lifespan)

@app.post("/generate")
async def generate_mesh_api(
    front: UploadFile = File(..., description="Front view image (required)"),
    back: UploadFile = File(None, description="Back view image (optional)"),
    left: UploadFile = File(None, description="Left view image (optional)"),
    right: UploadFile = File(None, description="Right view image (optional)")
):
    """
    Generate 3D mesh from uploaded images using pre-loaded models
    """
    
    # Create persistent temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Determine model version and process images
        if all(view is None for view in [back, left, right]):
            # Single image - use 2.1 model
            model_version = "2.1"
            front_path = os.path.join(temp_dir, "front.png")
            with open(front_path, "wb") as f:
                f.write(await front.read())
            image_input = front_path
        else:
            # Multi-view input
            model_version = "2mv" if "2mv" in pipeline_cache else "2.1"
            image_paths = {}
            
            # Save front image (required)
            front_path = os.path.join(temp_dir, "front.png")
            with open(front_path, "wb") as f:
                f.write(await front.read())
            image_paths["front"] = front_path
            
            # Save other views if provided
            view_files = {"back": back, "left": left, "right": right}
            
            for view_name, view_file in view_files.items():
                if view_file:
                    view_path = os.path.join(temp_dir, f"{view_name}.png")
                    with open(view_path, "wb") as f:
                        f.write(await view_file.read())
                    image_paths[view_name] = view_path
            
            image_input = image_paths
        
        # Generate mesh using cached models
        output_path = os.path.join(temp_dir, "output.glb")
        print(f"🎯 Generating mesh to: {output_path}")
        
        result_path = generate_3d_mesh_cached(image_input, output_path, model_version)
        
        # Verify the file exists and has content
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Generated mesh file not found: {result_path}")
        
        file_size = os.path.getsize(result_path)
        if file_size == 0:
            raise ValueError(f"Generated mesh file is empty: {result_path}")
        
        print(f"✅ Mesh file verified: {result_path} ({file_size} bytes)")
        
        # Create a response that cleans up after sending
        class CleanupFileResponse(FileResponse):
            def __init__(self, *args, cleanup_dir=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.cleanup_dir = cleanup_dir
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await super().__aexit__(exc_type, exc_val, exc_tb)
                if self.cleanup_dir and os.path.exists(self.cleanup_dir):
                    try:
                        shutil.rmtree(self.cleanup_dir)
                        print(f"🧹 Cleaned up temp directory: {self.cleanup_dir}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not clean up temp directory: {e}")
        
        return CleanupFileResponse(
            result_path,
            media_type="application/octet-stream",
            filename=f"generated_mesh_{model_version}.glb",
            cleanup_dir=temp_dir
        )
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"❌ Error in mesh generation: {e}")
        raise

@app.get("/models")
async def list_models():
    """List loaded models and their capabilities"""
    loaded_models = {}
    for version in pipeline_cache.keys():
        loaded_models[version] = {
            "name": f"Hunyuan3D-{version}",
            "description": "Latest single/multi-view model" if version == "2.1" else "Multi-view optimized model",
            "supported_views": ["front", "back", "left", "right"],
            "status": "loaded_in_gpu"
        }
    
    return {
        "loaded_models": loaded_models,
        "gpu_memory_usage": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else "N/A"
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Hunyuan3D Multi-Model API",
        "description": "Generate 3D meshes from single or multi-view images using pre-loaded models",
        "endpoints": {
            "/generate": "POST - Generate 3D mesh",
            "/models": "GET - List loaded models", 
            "/docs": "GET - API documentation"
        },
        "loaded_models": list(pipeline_cache.keys()),
        "status": "ready" if pipeline_cache else "loading"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(pipeline_cache),
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else "N/A"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)