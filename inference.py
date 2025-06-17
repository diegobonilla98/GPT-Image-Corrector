import torch
from PIL import Image
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
import os
import math
from model import PixelTransformerModel


def inference_image(model_path, input_image, output_image_path=None, num_pixels=128, input_dim=3, inference_batch_size=128, inference_stride=4):
    """
    Perform inference on a single image using the trained PixelTransformerModel.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        input_image (str or PIL.Image or numpy.ndarray): Path to the input image, or a PIL Image, or a numpy array
        output_image_path (str): Path to save the processed image
        num_pixels (int): Fallback number of pixels per sequence if not in model config.
        input_dim (int): Fallback input dimension (3 for RGB) if not in model config.
        inference_batch_size (int): Batch size for inference.
        inference_stride (int): Target stride. Primarily affects column stride.
                                Row stride is auto-adjusted for coverage, capped by this value.
                                Default is 4. Use 1 for densest processing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = device.type == 'cuda'

    print(f"Loading model from {model_path}...")
    if model_path.endswith('.pth') and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Use num_pixels from function args as a fallback
            model_config_num_pixels = num_pixels 
            
            if 'model_state_dict' in checkpoint:
                config = checkpoint.get('config', {})
                model_config_num_pixels = config.get('NUM_PIXELS', num_pixels)
                model = PixelTransformerModel(
                    num_pixels=model_config_num_pixels,
                    input_dim=config.get('INPUT_DIM', input_dim),
                    model_dim=config.get('MODEL_DIM', 256),
                    num_layers=config.get('NUM_LAYERS', 4),
                    nhead=config.get('NHEAD', 1),
                    dim_feedforward=config.get('DIM_FEEDFORWARD', 1024),
                    dropout=config.get('DROPOUT', 0.1)
                ).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            else: 
                model_config_num_pixels = num_pixels 
                model = PixelTransformerModel(
                    num_pixels=model_config_num_pixels,
                    input_dim=input_dim, # Use function arg input_dim
                    model_dim=256, 
                    num_layers=4,
                    nhead=1,
                    dim_feedforward=1024,
                    dropout=0.1
                ).to(device)
                model.load_state_dict(checkpoint)
                print("Loaded model from state dict")
            
            L_sequence_len = model.num_pixels # Use num_pixels from the loaded model

        except Exception as e:
            print(f"Error loading model: {e}. Check model_path and ensure parameters match saved model if loading state_dict directly.")
            return
    else:
        print(f"Model file not found: {model_path}")
        return
    
    transform = T.Compose([T.ToTensor()])
    
    try:
        if isinstance(input_image, str):
            img_pil = Image.open(input_image).convert('RGB')
        elif isinstance(input_image, np.ndarray):
            img_pil = Image.fromarray(input_image).convert('RGB')
        elif isinstance(input_image, Image.Image):
            img_pil = input_image.convert('RGB')
        else:
            raise ValueError("input_image must be a file path, numpy array, or PIL Image")
        
        print(f"Loaded image: {input_image}")
    except FileNotFoundError:
        print(f"Input image not found at {input_image}")
        return
        
    img_tensor_orig = transform(img_pil).to(device) 
    C_img, H, W = img_tensor_orig.shape # C_img is actual image channels
    print(f"Image dimensions: {C_img}x{H}x{W}")

    actual_model_input_dim = model.input_projection.in_features
    if C_img != actual_model_input_dim:
        print(f"Warning: Image channels {C_img} mismatch with model input_dim {actual_model_input_dim}")
        if C_img == 1 and actual_model_input_dim == 3:
            img_tensor_orig = img_tensor_orig.repeat(3, 1, 1)
            C_img = 3
            print("Converted image from Grayscale to RGB")
        elif C_img == 4 and actual_model_input_dim == 3:
            img_tensor_orig = img_tensor_orig[:3, :, :]
            C_img = 3
            print("Converted image from RGBA to RGB")
        else:
            print(f"Cannot reconcile channel mismatch {C_img} vs {actual_model_input_dim}. Exiting.")
            return
    
    # Ensure C for subsequent operations is the model's expected input dimension
    C = actual_model_input_dim

    model.eval()
    
    with torch.no_grad():
        N_total_pixels = H * W
        
        offsets = torch.arange(L_sequence_len, device=device)     
        img_pixels_value_flat = img_tensor_orig.permute(1, 2, 0).reshape(N_total_pixels, C)

        target_stride = max(1, inference_stride) # Ensure stride is at least 1

        if W == 0: # Avoid division by zero if width is 0
            print("Image width is 0. Cannot proceed.")
            return
            
        # Calculate how many rows a single sequence (starting at col 0) can span
        rows_spanned_by_one_sequence = math.ceil(L_sequence_len / W)
        
        # Determine actual_stride_H to ensure Y-axis coverage
        # It should be at most rows_spanned_by_one_sequence to ensure bands meet
        # It should also be at most the user's target_stride
        # And it must be at least 1
        max_permissible_H_stride = max(1, rows_spanned_by_one_sequence)
        actual_stride_H = min(target_stride, max_permissible_H_stride)
        actual_stride_H = max(1, actual_stride_H) # Ensure it's at least 1

        actual_stride_W = target_stride

        r_starts_strided = torch.arange(0, H, actual_stride_H, device=device, dtype=torch.long)
        c_starts_strided = torch.arange(0, W, actual_stride_W, device=device, dtype=torch.long)

        if len(r_starts_strided) == 0: r_starts_strided = torch.tensor([0], device=device, dtype=torch.long)
        if len(c_starts_strided) == 0: c_starts_strided = torch.tensor([0], device=device, dtype=torch.long)

        strided_start_coords_r, strided_start_coords_c = torch.meshgrid(r_starts_strided, c_starts_strided, indexing='ij')
        start_indices_flat = (strided_start_coords_r.flatten() * W + strided_start_coords_c.flatten())
        num_actual_sequences = len(start_indices_flat)
        
        print(f"Using L_sequence_len: {L_sequence_len}, Image W: {W}")
        print(f"Each sequence spans ~{rows_spanned_by_one_sequence} row(s).")
        print(f"Effective H stride: {actual_stride_H}, W stride: {actual_stride_W}.")
        print(f"Number of sequences to process: {num_actual_sequences} (Total pixels: {N_total_pixels})")


        seq_indices_flat_wrapped = (start_indices_flat.unsqueeze(1) + offsets.unsqueeze(0)) % N_total_pixels
        all_input_sequences = img_pixels_value_flat[seq_indices_flat_wrapped]

        sum_predictions_flat = torch.zeros(N_total_pixels, C, dtype=torch.float32, device=device)
        counts_flat = torch.zeros(N_total_pixels, dtype=torch.float32, device=device)
        
        print("Processing image...")
        pbar = tqdm(range(0, num_actual_sequences, inference_batch_size), desc="Inference Progress")
        for i in pbar:
            batch_input_seq = all_input_sequences[i : i + inference_batch_size]
            batch_indices_map = seq_indices_flat_wrapped[i : i + inference_batch_size]
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                predicted_batch_sequences = model(batch_input_seq)

            updates = predicted_batch_sequences.reshape(-1, C) 
            indices_to_update = batch_indices_map.reshape(-1)

            sum_predictions_flat.index_add_(0, indices_to_update, updates.float())
            
            ones_for_counts = torch.ones_like(indices_to_update, dtype=torch.float32, device=device)
            counts_flat.index_add_(0, indices_to_update, ones_for_counts)

        epsilon = 1e-8
        # Ensure counts_flat is broadcastable for division, handle 0 counts
        valid_counts = counts_flat.unsqueeze(1).clamp(min=epsilon)
        reconstructed_img_flat = sum_predictions_flat / valid_counts
        
        # For pixels that were never covered, counts_flat would be 0.
        # valid_counts makes them epsilon. sum_predictions is 0. So result is 0.
        # This is better than NaN. If you want interpolation for uncovered pixels,
        # that would be a more complex post-processing step.
        
        reconstructed_img_tensor = reconstructed_img_flat.reshape(H, W, C).permute(2, 0, 1)
        reconstructed_img_tensor = torch.clamp(reconstructed_img_tensor, 0.0, 1.0)
    
    to_pil = T.ToPILImage()
    reconstructed_img_pil = to_pil(reconstructed_img_tensor.cpu())
    
    if output_image_path is None:
        return reconstructed_img_pil
    reconstructed_img_pil.save(output_image_path)
    print(f"Processed image saved to: {output_image_path}")


if __name__ == '__main__':
    # Configuration
    MODEL_PATH = "checkpoints/best_model_epoch_26.pth"
    INPUT_IMAGE_PATH = r"G:\My Drive\PythonProjects\GPTImageCorrector\inputs\assets_task_01jxmcphjsfnx9gm2zjgbkczfb_1749809876_img_3.jpg"
    OUTPUT_IMAGE_PATH = "output.png"

    NUM_PIXELS_FALLBACK = 128 
    INPUT_DIM_FALLBACK = 3 
    INFERENCE_BATCH_SIZE = 128 
    # Try INFERENCE_STRIDE = 4 or 8.
    # If L_sequence_len is small relative to image width, row stride will become 1.
    # Column stride will be INFERENCE_STRIDE.
    INFERENCE_STRIDE = 4

    inference_image(
        model_path=MODEL_PATH,
        input_image_path=INPUT_IMAGE_PATH,
        output_image_path=OUTPUT_IMAGE_PATH,
        num_pixels=NUM_PIXELS_FALLBACK,
        input_dim=INPUT_DIM_FALLBACK,
        inference_batch_size=INFERENCE_BATCH_SIZE,
        inference_stride=INFERENCE_STRIDE
    )
