from inference import inference_image
from imwatermark import WatermarkEncoder
import gradio as gr
import torch
from PIL import Image
import numpy as np


# Constants
MODEL_PATH = "checkpoints/best_model_epoch_26.pth"
NUM_PIXELS_FALLBACK = 128
INPUT_DIM_FALLBACK = 3
WATERMARK_TEXT = "DBSw"

def get_max_batch_size():
    """Estimate max batch size for inference based on available GPU or CPU memory."""
    try:
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            # Reserve some memory for model weights and overhead, use ~1/8th for batch
            batch_size = int(total_mem // (8 * 1024 * 1024))  # ~1 batch per 8MB
            return max(16, min(batch_size, 1024))
        else:
            # For CPU, use a conservative default
            return 64
    except Exception:
        # Fallback in case of any error during CUDA check
        return 64

def blend_images(input_image, output_image, strength):
    """Blend input and output images based on strength value (0-100)."""
    if input_image is None or output_image is None:
        return None

    # Convert file path to PIL Image if needed
    if isinstance(input_image, str):
        input_image = Image.open(input_image).convert("RGB")
    if isinstance(output_image, str):
        output_image = Image.open(output_image).convert("RGB")

    # Convert to numpy arrays
    input_array = np.array(input_image)
    output_array = np.array(output_image)

    # Resize output to match input if necessary
    if input_array.shape != output_array.shape:
        output_image_resized = output_image.resize(input_image.size, Image.LANCZOS)
        output_array = np.array(output_image_resized)

    # Calculate blend ratio
    alpha = strength / 100.0

    # Weighted average
    blended_array = (1 - alpha) * input_array + alpha * output_array
    blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)

    return Image.fromarray(blended_array)

def run_inference(input_image, inference_batch_size, inference_stride, progress=gr.Progress()):
    if input_image is None:
        return None, gr.update(visible=False), gr.update(visible=False), None, None

    # Convert file path to PIL Image if needed
    if isinstance(input_image, str):
        input_image = Image.open(input_image).convert("RGB")
        
    progress(0, desc="🚀 Initializing AI model...")
        
    progress(0.1, desc="📸 Image saved, loading model...")
    
    # Call inference with progress tracking
    progress(0.2, desc="🧠 Running AI inference...")
    
    output_image = inference_image(
        model_path=MODEL_PATH,
        input_image=input_image,
        num_pixels=NUM_PIXELS_FALLBACK,
        input_dim=INPUT_DIM_FALLBACK,
        inference_batch_size=int(inference_batch_size),
        inference_stride=int(inference_stride)
    )
    
    progress(0.7, desc="🔍 Post-processing image...")
    # Add invisible watermark
    bgr = np.array(output_image)
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', WATERMARK_TEXT.encode('utf-8'))
    bgr_encoded = encoder.encode(bgr, 'dwtDct')
    output_image = Image.fromarray(bgr_encoded)

    progress(1.0, desc="🎉 Image correction complete!")
    
    # Create initial blend at 100% strength
    initial_blend = blend_images(input_image, output_image, 100)
    
    # Return all outputs at once
    return (
        output_image,                      # output_image
        gr.update(visible=True),           # blend_section visibility
        gr.update(visible=True),           # final_output_section visibility
        output_image,                      # corrected_image_state
        initial_blend                      # final_image
    )

# Custom CSS for styling
css = """
.title-container {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    margin-bottom: 2rem;
    border-radius: 15px;
    color: white;
}

.instruction-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
}

.input-column {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-right: 1rem;
}

.output-column {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-left: 1rem;
}

.blend-section {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-top: 1.5rem;
}

.control-panel {
    background: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 8px;
    backdrop-filter: blur(10px);
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as iface:
    
    # Title Section
    with gr.Row():
        gr.HTML("""
            <div class="title-container">
                <h1>🎨 GPT Image Corrector ✨</h1>
                <h3>Transform ChatGPT's Generated Images into Perfection!</h3>
            </div>
        """)
    
    # Instructions
    with gr.Row():
        gr.HTML("""
            <div class="instruction-box">
                <h3>🚀 How it Works:</h3>
                <p><strong>1.</strong> Upload an image generated by ChatGPT/DALL-E that needs correction</p>
                <p><strong>2.</strong> Adjust the processing parameters (stride controls quality vs speed)</p>
                <p><strong>3.</strong> Hit the magic "✨ Correct Image" button and watch the AI work!</p>
                <p><strong>4.</strong> Fine-tune the correction strength with the blend slider that appears after processing!</p>
                <p><em>💡 This AI model has been specially trained to fix common artifacts and improve image quality from AI-generated content.</em></p>
            </div>
        """)
    
    # Main Interface - Two Columns
    with gr.Row(equal_height=True):
        # Input Column
        with gr.Column(scale=1, elem_classes="input-column"): # Applied elem_classes
            # gr.HTML('<div class="input-column">') # Removed
            gr.Markdown("### 📤 Upload Your Image")
            input_image = gr.Image(
                type="filepath", 
                label="ChatGPT Generated Image",
                height=400
            )
            
            gr.Markdown("### ⚙️ Processing Settings")
            with gr.Row():
                stride = gr.Slider(
                    1, 8, 
                    value=4, 
                    step=1, 
                    label="🎯 Inference Stride",
                    info="Lower = Higher Quality (slower)"
                )
            
            batch_size = gr.Number(
                value=1024, 
                label="🔋 Batch Size",
                info="Adjust based on your hardware (default: 64)"
            )
            
            run_btn = gr.Button(
                "✨ Correct Image", 
                variant="primary",
                size="lg"
            )
            # gr.HTML('</div>') # Removed
        
        # Output Column  
        with gr.Column(scale=1, elem_classes="output-column"): # Applied elem_classes
            # gr.HTML('<div class="output-column">') # Removed
            gr.Markdown("### 📥 Corrected Result")
            output_image = gr.Image(
                type="pil", 
                label="AI-Corrected Image",
                height=400
            )
            
            gr.Markdown("### 📊 Processing Info")
            gr.HTML("""
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <p><strong>🎯 Model:</strong> best_model_epoch_26.pth</p>
                    <p><strong>🔧 Pixels:</strong> 128 | <strong>📐 Input Dim:</strong> 3</p>
                    <p><strong>⚡ Status:</strong> Ready to process your image!</p>
                </div>
            """)
            # gr.HTML('</div>') # Removed
    
    # Stage 2: Blend Control Section (Initially Hidden)
    with gr.Row(visible=False, elem_classes="blend-section") as blend_section: # Applied elem_classes
        # gr.HTML('<div class="blend-section">') # Removed
        gr.Markdown("### 🎛️ Fine-tune Correction Strength")
        
        with gr.Row():
            with gr.Column(scale=2):
                strength_slider = gr.Slider(
                    0, 100, 
                    value=100, 
                    step=1, 
                    label="🎯 Correction Strength",
                    info="0 = Original Image | 100 = Full Correction"
                )
            
            with gr.Column(scale=1):
                gr.HTML("""
                    <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                        <p><strong>💡 Tip:</strong> Adjust to find the perfect balance!</p>
                        <p><strong>🔄 Real-time:</strong> Changes apply instantly</p>
                    </div>
                """)
        
        # gr.HTML('</div>') # Removed
    
    # Final Blended Output (Initially Hidden)
    with gr.Row(visible=False) as final_output_section:
        with gr.Column():
            gr.Markdown("### 🎨 Final Result")
            final_image = gr.Image(
                type="pil", 
                label="Blended Image",
                height=400
            )
    
    # Store intermediate results for blending
    corrected_image_state = gr.State()
    
    # Event Handlers - Simplified to single handler
    run_btn.click(
        fn=run_inference,
        inputs=[input_image, batch_size, stride],
        outputs=[output_image, blend_section, final_output_section, corrected_image_state, final_image]
    )
    
    # Real-time blending on slider change
    strength_slider.change(
        fn=blend_images,
        inputs=[input_image, corrected_image_state, strength_slider],
        outputs=[final_image]
    )

if __name__ == "__main__":
    iface.queue().launch()
