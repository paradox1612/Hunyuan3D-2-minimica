from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import tempfile
import os
from generate_mv_enhanced import generate_3d_mesh

app = FastAPI(title="Hunyuan3D Multi-Model API")

@app.post("/generate")
async def generate_mesh_api(
    front: UploadFile = File(..., description="Front view image (required)"),
    back: UploadFile = File(None, description="Back view image (optional)"),
    left: UploadFile = File(None, description="Left view image (optional)"),
    right: UploadFile = File(None, description="Right view image (optional)")
):
    """
    Generate 3D mesh from uploaded images
    
    - **front**: Front view image (required)
    - **back**: Back view image (optional)
    - **left**: Left view image (optional) 
    - **right**: Right view image (optional)
    - **model_version**: Choose '2mv' for multi-view optimized or '2.1' for latest model
    
    For single image generation, only provide front image with model_version='2.1'
    For multi-view generation, provide 2 or more views
    """
    model_version = "2mv"
    with tempfile.TemporaryDirectory() as temp_dir:
        # Handle single image vs multi-view

        if all(view is None for view in [back, left, right]):
            # Single image for 2.1 model
            model_version = "2.1"
            front_path = os.path.join(temp_dir, "front.png")
            with open(front_path, "wb") as f:
                f.write(await front.read())
            image_input = front_path
        else:
            # Multi-view input
            image_paths = {}
            
            # Save front image (required)
            front_path = os.path.join(temp_dir, "front.png")
            with open(front_path, "wb") as f:
                f.write(await front.read())
            image_paths["front"] = front_path
            
            # Save other views if provided
            view_files = {
                "back": back,
                "left": left, 
                "right": right
            }
            
            for view_name, view_file in view_files.items():
                if view_file:
                    view_path = os.path.join(temp_dir, f"{view_name}.png")
                    with open(view_path, "wb") as f:
                        f.write(await view_file.read())
                    image_paths[view_name] = view_path
            
            image_input = image_paths
        
        # Generate mesh
        output_path = os.path.join(temp_dir, "output.glb")
        generate_3d_mesh(image_input, output_path, model_version)
        
        # Return file
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"generated_mesh_{model_version}.glb"
        )

@app.get("/models")
async def list_models():
    """List available models and their capabilities"""
    return {
        "models": {
            "2mv": {
                "name": "Hunyuan3D-2mv",
                "description": "Multi-view optimized model",
                "supported_views": ["front", "back", "left", "right"],
                "min_views": 1,
                "optimal_views": 3
            },
            "2.1": {
                "name": "Hunyuan3D-2.1", 
                "description": "Latest single/multi-view model",
                "supported_views": ["front", "back", "left", "right"],
                "min_views": 1,
                "optimal_views": "1 for single-image, 3+ for multi-view"
            }
        }
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Hunyuan3D Multi-Model API",
        "description": "Generate 3D meshes from single or multi-view images",
        "endpoints": {
            "/generate": "POST - Generate 3D mesh",
            "/models": "GET - List available models", 
            "/docs": "GET - API documentation"
        },
        "supported_views": ["front", "back", "left", "right"],
        "supported_formats": ["PNG", "JPG", "JPEG"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)