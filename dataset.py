from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import math


class PixelDataset(Dataset):
    """
    A dataset that loads pixels from input-output image pairs.

    Args:
        root_dir (str): Path to the root directory containing 'input' and 'output' subfolders.
        transform (callable, optional): Optional transform to be applied to the images
            before pixel extraction. Expected to convert PIL Image to Tensor.
            If None, a default ToTensor transform is applied.
        mode (str): Pixel loading mode. One of "1D", "2D", or "RAND".
            - "1D": Pixels are loaded as consecutive pixel lines with warping, length num_pixels.
            - "2D": Pixels are loaded as a sqrt(num_pixels) x sqrt(num_pixels) patch.
            - "RAND": num_pixels pixels are randomly sampled from the input/output pair.
            Defaults to "1D".
        num_pixels (int): Number of pixels to extract for each sample.
            For "2D" mode, this must be a perfect square. Defaults to 64.
        stride_length (int): Stride length for "1D" and "2D" modes when defining samples.
            Defaults to 1.
    """
    def __init__(self, input_image_paths_list, output_image_paths_list, transform=None, mode="1D", num_pixels=64, stride_length=1):
        self.mode = mode.upper()
        self.num_pixels = num_pixels
        self.stride_length = stride_length
        
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        if self.mode not in ["1D", "2D", "RAND"]:
            raise ValueError("mode must be '1D', '2D', or 'RAND'")

        if self.mode == "2D":
            sqrt_num_pixels = math.sqrt(self.num_pixels)
            if not sqrt_num_pixels.is_integer():
                raise ValueError("For '2D' mode, num_pixels must be a perfect square.")
            self.kernel_dim = int(sqrt_num_pixels)

        self.input_image_paths = input_image_paths_list
        self.output_image_paths = output_image_paths_list

        if not self.input_image_paths or not self.output_image_paths:
            raise FileNotFoundError("No images found in input/output subfolders.")
        if len(self.input_image_paths) != len(self.output_image_paths):
            raise ValueError("Mismatch between number of input and output images.")

        self._buffer_pixels = []
        self._image_dims_cache = {} # Cache image dimensions: {img_idx: (W, H, C_tensor)}

        for img_idx in range(len(self.input_image_paths)):
            # Get image dimensions by loading one image and transforming it
            # This is to ensure we use dimensions after transformation (e.g. if ToTensor changes H,W order)
            # For simplicity, we assume input and output images have same dimensions after transform
            if img_idx not in self._image_dims_cache:
                try:
                    temp_img_pil = Image.open(self.input_image_paths[img_idx]).convert('RGB')
                    temp_tensor = self.transform(temp_img_pil)
                    # Assuming tensor is C x H x W
                    C_tensor, H_tensor, W_tensor = temp_tensor.shape
                    self._image_dims_cache[img_idx] = (W_tensor, H_tensor, C_tensor)
                except Exception as e:
                    raise IOError(f"Could not load or transform image {self.input_image_paths[img_idx]}: {e}")

            W, H, _ = self._image_dims_cache[img_idx]

            if self.mode == "1D":
                for r_start in range(0, H, self.stride_length):
                    for c_start in range(0, W, self.stride_length):
                        self._buffer_pixels.append((img_idx, r_start, c_start))
            elif self.mode == "2D":
                if H >= self.kernel_dim and W >= self.kernel_dim:
                    for r_start in range(0, H - self.kernel_dim + 1, self.stride_length):
                        for c_start in range(0, W - self.kernel_dim + 1, self.stride_length):
                            self._buffer_pixels.append((img_idx, r_start, c_start))
            elif self.mode == "RAND":
                # For RAND mode, each image pair is one item in _buffer_pixels
                # The random sampling happens in __getitem__
                # To make len(dataset) reflect number of images:
                if not any(item[0] == img_idx for item in self._buffer_pixels): # Add once per image
                     self._buffer_pixels.append((img_idx,))

    def __len__(self):
        return len(self._buffer_pixels)

    def __getitem__(self, idx):
        item_info = self._buffer_pixels[idx]
        img_idx = item_info[0]

        # Load and transform images
        input_img_pil = Image.open(self.input_image_paths[img_idx]).convert('RGB')
        output_img_pil = Image.open(self.output_image_paths[img_idx]).convert('RGB')

        input_tensor = self.transform(input_img_pil)
        output_tensor = self.transform(output_img_pil)
        
        C, H, W = input_tensor.shape 

        if self.mode == "1D":
            _, r_start, c_start = item_info
            
            input_pixels_collected = torch.empty((self.num_pixels, C), dtype=input_tensor.dtype)
            output_pixels_collected = torch.empty((self.num_pixels, C), dtype=output_tensor.dtype)

            current_r, current_c = r_start, c_start
            for i in range(self.num_pixels):
                input_pixels_collected[i] = input_tensor[:, current_r, current_c]
                output_pixels_collected[i] = output_tensor[:, current_r, current_c]
                
                current_c += 1
                if current_c >= W:
                    current_c = 0
                    current_r += 1
                    if current_r >= H:
                        current_r = 0 # Warping

            input_final = input_pixels_collected
            output_final = output_pixels_collected

        elif self.mode == "2D":
            _, r_start, c_start = item_info
            input_patch = input_tensor[:, r_start : r_start + self.kernel_dim, c_start : c_start + self.kernel_dim]
            output_patch = output_tensor[:, r_start : r_start + self.kernel_dim, c_start : c_start + self.kernel_dim]
            C = input_tensor.shape[0] # Number of channels
            input_final = input_patch.contiguous().view(C, -1).permute(1, 0) # Shape: (kernel_dim*kernel_dim, C)
            output_final = output_patch.contiguous().view(C, -1).permute(1, 0) # Shape: (kernel_dim*kernel_dim, C)

        elif self.mode == "RAND":
            rand_rows = torch.randint(0, H, size=(self.num_pixels,))
            rand_cols = torch.randint(0, W, size=(self.num_pixels,))
            
            # input_tensor is C, H, W. Accessing with [:, rand_rows, rand_cols] gives C, num_pixels
            input_final = input_tensor[:, rand_rows, rand_cols].permute(1, 0) # num_pixels, C
            output_final = output_tensor[:, rand_rows, rand_cols].permute(1, 0) # num_pixels, C
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return input_final, output_final
