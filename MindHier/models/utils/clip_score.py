import torch
import torch.nn as nn
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from typing import Union, List, Dict, Any
from PIL import Image # For type hinting images

class FrozenCLIPEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text and vision (from huggingface).
    Includes methods to encode text and images, returning projected embeddings and hidden states.
    """

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device=None,
        max_length=77,
        freeze=True,
        dtype=torch.float32,
        cache_dir="pretrained/clip",
        local_files_only=False,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            version, cache_dir=cache_dir, local_files_only=local_files_only
        )
        # Load models with the specified dtype and ...WithProjection variants
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            version, cache_dir=cache_dir, local_files_only=local_files_only
        ).to(device=device, dtype=dtype)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            version, cache_dir=cache_dir, local_files_only=local_files_only
        ).to(device=device, dtype=dtype)
        # Add the image processor for preprocessing image inputs
        self.image_processor = CLIPImageProcessor.from_pretrained(
            version, cache_dir=cache_dir, local_files_only=local_files_only
        )

        self.device = device
        self.dtype = dtype 
        self.max_length = max_length

        # The hidden dimensionality of the text transformer's layers (pre-projection).
        self.text_model_hidden_size = self.text_encoder.config.hidden_size
        # The hidden dimensionality of the vision transformer's layers (pre-projection).
        self.image_model_hidden_size = self.image_encoder.config.hidden_size
        # The dimensionality of the final projected embeddings (shared space).
        self.projection_dim = self.text_encoder.config.projection_dim
        
        if freeze:
            self.freeze()

    def freeze(self):
        self.text_encoder = self.text_encoder.eval()
        self.image_encoder = self.image_encoder.eval()
        
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Encodes text using the CLIPTextModelWithProjection.
        Args:
            text (Union[str, List[str]]): The input text or list of texts to encode.
        Returns:
            dict: A dictionary containing:
                - "text_embeds": torch.Tensor of shape (batch_size, projection_dim) - The projected text embedding.
                - "last_hidden_state": torch.Tensor of shape (batch_size, sequence_length, text_hidden_size)
                - "attn_bias": torch.Tensor for attention masking.
        """
        if isinstance(text, str):
            text = [text] 

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = batch_encoding.input_ids.to(self.device)
        attention_mask = batch_encoding.attention_mask.to(self.device)

        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True, 
            output_attentions=False 
        )

        text_embeds = text_outputs.text_embeds
        last_hidden_state = text_outputs.last_hidden_state
        
        # Construct the attention bias
        # This bias can be used by subsequent attention mechanisms if needed
        attn_bias = attention_mask.to(last_hidden_state.dtype) # Ensure dtype consistency
        attn_bias[attn_bias == 0] = -torch.inf # Or a very large negative number like -1e9
        attn_bias[attn_bias == 1] = 0.0

        return {
            "text_embeds": text_embeds,
            "last_hidden_state": last_hidden_state,
            "attn_bias": attn_bias
        }

    @torch.no_grad()
    def encode_image(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        Encodes images using the CLIPVisionModelWithProjection.
        Args:
            images (Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor]): 
                The input image(s). Can be a single PIL image, a list of PIL images,
                or a preprocessed tensor of shape (batch_size, num_channels, height, width).
        Returns:
            torch.Tensor: The projected image embeddings of shape (batch_size, projection_dim).
        """
        if isinstance(images, Image.Image): # Single PIL Image
            images = [images]
        
        # If images are already tensors, assume they are correctly preprocessed.
        # Otherwise, use the image_processor.
        if not isinstance(images, torch.Tensor):
            # Process PIL images
            # The image_processor handles conversion to tensors, normalization, etc.
            # It returns a dict-like object (BatchFeature) containing 'pixel_values'.
            processed_inputs = self.image_processor(images, return_tensors="pt")
            pixel_values = processed_inputs.pixel_values
        else:
            # Assume input is already a correctly shaped tensor
            pixel_values = images

        # Ensure pixel_values are on the correct device and dtype
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        image_outputs = self.image_encoder(
            pixel_values=pixel_values,
            output_hidden_states=False, # Not typically needed for image_embeds
            output_attentions=False
        )
        
        # The projected image embedding
        return image_outputs.image_embeds


class CLIPScoreCalculator(nn.Module):
    """
    Calculates one-to-one CLIP similarity scores between two batches of images
    using a provided FrozenCLIPEmbedder instance.
    """
    def __init__(self, clip_embedder: FrozenCLIPEmbedder):
        super().__init__()
        if not isinstance(clip_embedder, FrozenCLIPEmbedder):
            raise TypeError("clip_embedder must be an instance of FrozenCLIPEmbedder.")
        self.clip_embedder = clip_embedder
        # The logit_scale is learned during CLIP's pre-training.
        # For calculating similarity scores (cosine similarity), it's often not applied,
        # or if applied, it's usually to match the logits from the original CLIP paper (logit_scale.exp()).
        # Here, we will return the raw cosine similarity.
        # self.logit_scale = clip_embedder.text_encoder.logit_scale # If accessible and needed

    @torch.no_grad()
    def forward(self, 
                images1: Union[List[Image.Image], torch.Tensor], 
                images2: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        Calculates one-to-one CLIP similarity scores between two batches of images.
        Args:
            images1 (Union[List[PIL.Image.Image], torch.Tensor]): The first batch of images.
            images2 (Union[List[PIL.Image.Image], torch.Tensor]): The second batch of images.
                                                                Must be the same length as images1.
        Returns:
            torch.Tensor: A 1D tensor of cosine similarity scores of shape (batch_size,),
                          one for each corresponding pair of images.
        """
        if not hasattr(images1, '__len__') or not hasattr(images2, '__len__'):
             raise TypeError("Inputs images1 and images2 must be sequences (e.g., lists or tensors).")
        if len(images1) != len(images2):
            raise ValueError("Input batches images1 and images2 must have the same length "
                             "for one-to-one comparison.")
        if len(images1) == 0:
            return torch.tensor([], device=self.clip_embedder.device, dtype=self.clip_embedder.dtype)
        # Encode images to get their features (projected embeddings)
        features1 = self.clip_embedder.encode_image(images1)  # Shape: (batch_size, projection_dim)
        features2 = self.clip_embedder.encode_image(images2)  # Shape: (batch_size, projection_dim)

        # Normalize features to unit vectors
        features1_norm = features1 / features1.norm(dim=-1, keepdim=True)
        features2_norm = features2 / features2.norm(dim=-1, keepdim=True)

        # Calculate element-wise cosine similarity for each pair
        # (N, D) * (N, D) -> sum over D -> (N)
        # This computes dot product of corresponding normalized vectors.
        similarity_scores = (features1_norm * features2_norm).sum(dim=-1)
        
        return similarity_scores
    
    
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import ImageDraw, ImageFont
from typing import Union, Tuple, List

class ScoreTextImageOverlay:
    """
    A class to draw scores as text overlay onto a batch of images.
    Each image in the batch is processed individually.
    """

    def __init__(self,
                 font_path: str = "arial.ttf",
                 font_size: int = 20,
                 text_color: Union[str, Tuple[int, int, int]] = "red",
                 text_xy: Tuple[int, int] = (5, 5),
                 score_format: str = "Score: {:.3f}"):
        """
        Initializes the ScoreTextImageOverlay.

        Args:
            font_path (str): Path to the .ttf font file. "arial.ttf" is a common default.
            font_size (int): Desired size of the font.
            text_color (Union[str, Tuple[int, int, int]]): Color of the text.
                Can be a color name (e.g., "red", "yellow") or an RGB tuple (e.g., (255, 255, 0)).
            text_xy (Tuple[int, int]): (x, y) coordinates for the top-left position of the text
                on each individual image.
            score_format (str): Python f-string format for displaying the score.
        """
        self.text_color = text_color
        self.text_xy = text_xy
        self.score_format = score_format

        try:
            self.font = ImageFont.truetype(font_path, font_size)
        except IOError:
            # print(f"Warning: Font '{font_path}' at size {font_size} not found or cannot be opened. "
            #       "Using default system font.")
            self.font = ImageFont.load_default()
            # print("Note: Default font does not use 'font_size' parameter; text size will be fixed and may be small.")
        except Exception as e:
            print(f"An unexpected error occurred while loading font '{font_path}': {e}. Using default font.")
            self.font = ImageFont.load_default()


    @torch.no_grad()  # Operations in this method should not affect gradients
    def forward(self,
                image_batch_tensor: torch.Tensor,
                scores_tensor: torch.Tensor) -> torch.Tensor:
        """
        Adds scores as text overlay to each image in the input batch.
        The input `image_batch_tensor` is assumed to contain images where
        two original images have already been concatenated (e.g., vertically).

        Args:
            image_batch_tensor (torch.Tensor): A batch of images with shape (N, C, H, W).
                Tensor values can be in any range ToPILImage accepts (e.g., [0,1] or [-1,1]).
            scores_tensor (torch.Tensor): A 1D tensor of scores (N,), one for each image.

        Returns:
            torch.Tensor: A new batch of images (N, C, H, W) with scores drawn on them.
                          The output tensor values will be in the range [0.0, 1.0].
                          The device of the output tensor will match the input image_batch_tensor.
        """
        if not isinstance(image_batch_tensor, torch.Tensor) or not isinstance(scores_tensor, torch.Tensor):
            raise TypeError("Inputs image_batch_tensor and scores_tensor must be PyTorch tensors.")
        if image_batch_tensor.ndim != 4:
            raise ValueError(f"Expected image_batch_tensor to be 4D (N, C, H, W), got {image_batch_tensor.ndim}D.")
        if scores_tensor.ndim != 1:
            raise ValueError(f"Expected scores_tensor to be 1D (N,), got {scores_tensor.ndim}D.")
        if image_batch_tensor.shape[0] != scores_tensor.shape[0]:
            raise ValueError(
                f"Batch size mismatch: {image_batch_tensor.shape[0]} images and {scores_tensor.shape[0]} scores."
            )

        if image_batch_tensor.shape[0] == 0: # Handle empty batch
            return torch.empty_like(image_batch_tensor)

        # Detach and move tensors to CPU for PIL operations
        images_cpu = image_batch_tensor.cpu().detach()
        scores_cpu = scores_tensor.cpu().detach()

        processed_images_list: List[torch.Tensor] = []
        to_pil = ToPILImage()
        # ToTensor converts PIL images (mode L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, HSL, HSV)
        # in range [0, 255] to Tensors of shape (C, H, W) in range [0.0, 1.0].
        to_tensor = ToTensor()

        for i in range(images_cpu.shape[0]):
            single_image_tensor = images_cpu[i]  # Shape (C, H, W)
            score = scores_cpu[i].item()

            # Convert tensor to PIL Image. ToPILImage handles various input tensor value ranges.
            pil_image = to_pil(single_image_tensor)

            # Ensure image is in RGB mode for colored text, if it's not (e.g. grayscale)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            draw = ImageDraw.Draw(pil_image)
            score_text = self.score_format.format(score)
            draw.text(self.text_xy, score_text, fill=self.text_color, font=self.font)

            # Convert modified PIL image back to a tensor. Output range is [0.0, 1.0].
            tensor_with_text = to_tensor(pil_image)
            processed_images_list.append(tensor_with_text)

        # Stack processed images back into a batch and move to the original device
        output_batch_tensor = torch.stack(processed_images_list).to(image_batch_tensor.device)

        return output_batch_tensor
