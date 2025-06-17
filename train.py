import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from PIL import Image
import glob
import os
from tqdm import tqdm
import numpy as np

from dataset import PixelDataset
from model import PixelTransformerModel


CONFIG = {
    "NUM_EPOCHS": 50,
    "BATCH_SIZE": 256,
    "PEAK_LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 0.01,
    "MODEL_DIM": 256,
    "NUM_LAYERS": 4,
    "NHEAD": 1,
    "DIM_FEEDFORWARD": 1024,
    "DROPOUT": 0.1,
    "NUM_PIXELS": 128,
    "INPUT_DIM": 3,
    "DATASET_MODE": "1D",
    "DATASET_STRIDE": 64,
    "WARMUP_STEPS": 4000,

    "INPUT_DIR_TRAIN": "inputs",
    "OUTPUT_DIR_TRAIN": "outputs",

    "REFERENCE_IMAGE_PATH": "./assetstask_01jxpxvq0mf8z9jhgtqz58t24h1749895070_img_0.png",
    "REF_IMG_INFERENCE_BATCH_SIZE": 128,

    "CHECKPOINT_DIR": "checkpoints",
    "TENSORBOARD_LOG_DIR": "runs/pixel_transformer_experiment",
    "NUM_WORKERS": min(os.cpu_count(), 4),
    "VAL_SPLIT": 0.1,  # Fraction of data to use for validation
}

def get_image_paths(input_dir, output_dir, extensions=("*.png", "*.jpg", "*.jpeg")):
    input_paths = []
    output_paths = []
    # Collect all input files with supported extensions
    input_files = []
    for ext in extensions:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))
    input_files = sorted(input_files)
    # Collect all output files with supported extensions
    output_files = []
    for ext in extensions:
        output_files.extend(glob.glob(os.path.join(output_dir, ext)))
    output_files = sorted(output_files)
    # Build a mapping from output basenames (without extension) to full path (keep only the first found)
    output_map = {}
    for f in output_files:
        base = os.path.splitext(os.path.basename(f))[0]
        if base not in output_map:
            output_map[base] = f
    for in_file in input_files:
        base = os.path.splitext(os.path.basename(in_file))[0]
        out_file = output_map.get(base)
        if out_file and os.path.exists(out_file):
            input_paths.append(in_file)
            output_paths.append(out_file)
        else:
            print(f"Warning: Output file for input {in_file} not found in {output_dir}")
    return input_paths, output_paths

def split_dataset(input_paths, output_paths, val_split=0.1, seed=42):
    assert len(input_paths) == len(output_paths)
    indices = list(range(len(input_paths)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    val_size = int(len(indices) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_input = [input_paths[i] for i in train_indices]
    train_output = [output_paths[i] for i in train_indices]
    val_input = [input_paths[i] for i in val_indices]
    val_output = [output_paths[i] for i in val_indices]
    return train_input, train_output, val_input, val_output

# Learning Rate Scheduler (Vaswani et al., 2017 "Attention Is All You Need")
def get_lr_lambda(warmup_steps, model_dim, peak_lr_scale_factor):
    def lr_lambda_fn(current_step):
        current_step += 1
        arg1 = current_step ** -0.5
        arg2 = current_step * (warmup_steps ** -1.5)
        return peak_lr_scale_factor * (warmup_steps ** 0.5) * min(arg1, arg2)
        
    return lr_lambda_fn

def process_reference_image(model, image_path, num_pixels, input_dim, device, transform, inference_batch_size):
    model.eval()
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Reference image not found at {image_path}. Skipping reference image processing.")
        return None, None
        
    img_tensor_orig = transform(img_pil).to(device) # Shape: (C, H, W)
    C, H, W = img_tensor_orig.shape

    if C != input_dim:
        print(f"Warning: Reference image channel {C} mismatch with model input_dim {input_dim}. Attempting to proceed.")
        # Potentially add more robust handling or ensure transform handles this.
        # For now, we assume the model's input_dim is what we should work with for sequence channels.
        if C == 1 and input_dim == 3: # Grayscale to RGB
            img_tensor_orig = img_tensor_orig.repeat(3,1,1)
            C = 3
            print("Converted reference image from Grayscale to RGB.")
        elif C == 4 and input_dim == 3: # RGBA to RGB
            img_tensor_orig = img_tensor_orig[:3,:,:]
            C = 3
            print("Converted reference image from RGBA to RGB.")
        elif C != input_dim:
             print(f"Cannot automatically reconcile channel mismatch {C} vs {input_dim}. Skipping ref image.")
             return img_tensor_orig, None # Return original if processing fails due to channels


    with torch.no_grad():
        # 1. Prepare for sequence generation
        N_total_pixels = H * W
        L_sequence_len = num_pixels

        # Create indices for all starting pixels (0 to N-1)
        start_indices = torch.arange(N_total_pixels, device=device) # Shape: (N_total_pixels)
        # Create offsets for pixels within a sequence (0 to L-1)
        offsets = torch.arange(L_sequence_len, device=device)     # Shape: (L_sequence_len)

        # Generate 1D indices for all pixels in all sequences with wrapping
        # seq_indices_flat_wrapped[i, j] = (start_indices[i] + offsets[j]) % N_total_pixels
        seq_indices_flat_wrapped = (start_indices.unsqueeze(1) + offsets.unsqueeze(0)) % N_total_pixels
        # Shape: (N_total_pixels, L_sequence_len)

        # 2. Prepare flat pixel data from the original image
        # Reshape image from (C, H, W) to (N_total_pixels, C)
        img_pixels_value_flat = img_tensor_orig.permute(1, 2, 0).reshape(N_total_pixels, C)
        # Shape: (N_total_pixels, C)

        # 3. Generate all input sequences using advanced indexing (gather operation)
        # all_sequences[i,j,k] will be the k-th channel of the j-th pixel in the i-th sequence
        all_input_sequences = img_pixels_value_flat[seq_indices_flat_wrapped]
        # Shape: (N_total_pixels, L_sequence_len, C)
        # N_total_pixels here is the number of sequences (one starting at each pixel)

        # 4. Initialize accumulators (flat)
        sum_predictions_flat = torch.zeros(N_total_pixels, C, dtype=torch.float32, device=device)
        counts_flat = torch.zeros(N_total_pixels, dtype=torch.float32, device=device)

        # 5. Process in batches
        num_sequences_to_process = N_total_pixels
        
        ref_img_pbar = tqdm(range(0, num_sequences_to_process, inference_batch_size), desc="Processing Ref Img", leave=False)
        for i in ref_img_pbar:
            batch_input_seq = all_input_sequences[i : i + inference_batch_size]
            # These are the 1D indices in the original flat image that correspond to each pixel in batch_input_seq
            batch_indices_map = seq_indices_flat_wrapped[i : i + inference_batch_size]
            
            # Get model predictions for the batch
            predicted_batch_sequences = model(batch_input_seq) # Shape: (current_batch_size, L_sequence_len, C)

            # Flatten predictions and corresponding indices for index_add_
            updates = predicted_batch_sequences.reshape(-1, C) # Shape: (current_batch_size * L_sequence_len, C)
            indices_to_update = batch_indices_map.reshape(-1)  # Shape: (current_batch_size * L_sequence_len)

            # Accumulate predictions
            sum_predictions_flat.index_add_(0, indices_to_update, updates)
            
            # Increment counts for affected pixels
            ones_for_counts = torch.ones_like(indices_to_update, dtype=torch.float32, device=device)
            counts_flat.index_add_(0, indices_to_update, ones_for_counts)

        # 6. Finalize: Average predictions and reshape back to image dimensions
        epsilon = 1e-8
        # Ensure counts_flat is broadcastable for division: (N_total_pixels, 1)
        reconstructed_img_flat = sum_predictions_flat / (counts_flat.unsqueeze(1) + epsilon)
        
        # Reshape from (N_total_pixels, C) back to (C, H, W)
        reconstructed_img_tensor = reconstructed_img_flat.reshape(H, W, C).permute(2, 0, 1)
        reconstructed_img_tensor = torch.clamp(reconstructed_img_tensor, 0.0, 1.0)
    
    return img_tensor_orig, reconstructed_img_tensor

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
    os.makedirs(CONFIG["TENSORBOARD_LOG_DIR"], exist_ok=True)

    transform = T.Compose([
        T.ToTensor(),
    ])

    input_paths, output_paths = get_image_paths(CONFIG["INPUT_DIR_TRAIN"], CONFIG["OUTPUT_DIR_TRAIN"])
    if not input_paths:
        print(f"No images found in {CONFIG['INPUT_DIR_TRAIN']} and {CONFIG['OUTPUT_DIR_TRAIN']}. Exiting.")
        return

    train_input_paths, train_output_paths, val_input_paths, val_output_paths = split_dataset(
        input_paths, output_paths, val_split=CONFIG["VAL_SPLIT"]
    )

    if not train_input_paths:
        print("No training images after split. Exiting.")
        return
    if not val_input_paths:
        print("No validation images after split. Validation will be skipped.")

    train_dataset = PixelDataset(
        input_image_paths_list=train_input_paths,
        output_image_paths_list=train_output_paths,
        transform=transform,
        mode=CONFIG["DATASET_MODE"],
        num_pixels=CONFIG["NUM_PIXELS"],
        stride_length=CONFIG["DATASET_STRIDE"]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=True if device.type == 'cuda' else False
    )

    val_dataloader = None
    if val_input_paths:
        val_dataset = PixelDataset(
            input_image_paths_list=val_input_paths,
            output_image_paths_list=val_output_paths,
            transform=transform,
            mode=CONFIG["DATASET_MODE"],
            num_pixels=CONFIG["NUM_PIXELS"],
            stride_length=CONFIG["DATASET_STRIDE"]
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=False,
            num_workers=CONFIG["NUM_WORKERS"],
            pin_memory=True if device.type == 'cuda' else False
        )

    # --- Model, Optimizer, Loss, Scheduler ---
    model = PixelTransformerModel(
        num_pixels=CONFIG["NUM_PIXELS"],
        input_dim=CONFIG["INPUT_DIM"],
        model_dim=CONFIG["MODEL_DIM"],
        num_layers=CONFIG["NUM_LAYERS"],
        nhead=CONFIG["NHEAD"],
        dim_feedforward=CONFIG["DIM_FEEDFORWARD"],
        dropout=CONFIG["DROPOUT"]
    ).to(device)

    criterion = nn.MSELoss() # Or nn.L1Loss()
    # Optimizer initial LR is 1.0; scheduler will provide the actual LR scaling factor.
    optimizer = optim.AdamW(model.parameters(), lr=1.0, weight_decay=CONFIG["WEIGHT_DECAY"])
    
    lr_scheduler_lambda_fn = get_lr_lambda(
        warmup_steps=CONFIG["WARMUP_STEPS"], 
        model_dim=CONFIG["MODEL_DIM"], # Not directly used in my simplified lambda, but part of original formula
        peak_lr_scale_factor=CONFIG["PEAK_LEARNING_RATE"]
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler_lambda_fn)

    writer = SummaryWriter(CONFIG["TENSORBOARD_LOG_DIR"])
    best_val_loss = float('inf')
    global_step = 0

    print("Starting training...")
    for epoch in range(CONFIG["NUM_EPOCHS"]):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} [Train]", leave=False)
        
        for batch_idx, (input_pixels, target_pixels) in enumerate(train_progress_bar):
            input_pixels = input_pixels.to(device)   # (B, num_pixels, C)
            target_pixels = target_pixels.to(device) # (B, num_pixels, C)

            optimizer.zero_grad()
            predictions = model(input_pixels)
            loss = criterion(predictions, target_pixels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            scheduler.step() # Scheduler step per batch/iteration

            epoch_train_loss += loss.item()
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_step)
            train_progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            global_step += 1
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- Validation Phase ---
        if val_dataloader:
            model.eval()
            epoch_val_loss = 0
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} [Val]", leave=False)
            with torch.no_grad():
                for input_pixels, target_pixels in val_progress_bar:
                    input_pixels = input_pixels.to(device)
                    target_pixels = target_pixels.to(device)
                    predictions = model(input_pixels)
                    loss = criterion(predictions, target_pixels)
                    epoch_val_loss += loss.item()
                    val_progress_bar.set_postfix(loss=loss.item())
            
            avg_val_loss = epoch_val_loss / len(val_dataloader)
            writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(CONFIG["CHECKPOINT_DIR"], f"best_model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_val_loss,
                    'config': CONFIG
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        else: # If no validation, save model periodically or last one
            if (epoch + 1) % 10 == 0 or (epoch + 1) == CONFIG["NUM_EPOCHS"]: # Save every 10 epochs or last
                 checkpoint_path = os.path.join(CONFIG["CHECKPOINT_DIR"], f"model_epoch_{epoch+1}.pth")
                 torch.save(model.state_dict(), checkpoint_path)
                 print(f"Saved model checkpoint to {checkpoint_path}")


        # --- Process and Log Reference Image ---
        if CONFIG["REFERENCE_IMAGE_PATH"]:
            print(f"Processing reference image for epoch {epoch+1}...")
            original_ref_img, processed_ref_img = process_reference_image(
                model,
                CONFIG["REFERENCE_IMAGE_PATH"],
                CONFIG["NUM_PIXELS"],
                CONFIG["INPUT_DIM"],
                device,
                transform, # Use the same ToTensor transform
                CONFIG["REF_IMG_INFERENCE_BATCH_SIZE"]
            )
            if original_ref_img is not None and processed_ref_img is not None:
                # Add images to TensorBoard: expects (N, C, H, W) or (C, H, W)
                writer.add_image('Reference/Original', original_ref_img, epoch)
                writer.add_image('Reference/Processed', processed_ref_img, epoch)
                print("Reference image processed and logged to TensorBoard.")
            else:
                print("Skipped logging reference image due to processing error or file not found.")


    writer.close()
    print("Training finished.")

if __name__ == '__main__': 
    train_model()
