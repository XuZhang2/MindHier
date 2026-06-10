import torch
import clip
from torchvision import transforms
import numpy as np
import torch.nn as nn
from accelerate import Accelerator, DeepSpeedPlugin
import os, time, logging
import torch.nn.functional as F
import matplotlib.pyplot as plt


logs = set()

def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def init_tensorboard():
    from torch.utils.tensorboard import SummaryWriter
    log_dir = './logs/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def config_multi_gpu():
    # Multi-GPU config
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)
    accelerator = Accelerator(split_batches=False, mixed_precision='no', deepspeed_plugin=deepspeed_plugin)  
    accelerator.print("PID of this process =",os.getpid())
    device = accelerator.device
    accelerator.print("device:",device)
    num_devices = torch.cuda.device_count()
    if num_devices==0: num_devices = 1
    accelerator.print(accelerator.state)
    local_rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == 'NO'
    accelerator.print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)

    return accelerator, device, local_rank


def prepare_scheduler(volxel_encoder, cfg):

    params = list(volxel_encoder.parameters())
    optimizer = torch.optim.AdamW(params, betas=(0.9, 0.9999), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, eps=1e-8)

    num_devices = max(torch.cuda.device_count(), 1)
    global_batch_size = cfg.train.batch_size * num_devices
    total_steps=int(cfg.train.epochs * ( 67626 // global_batch_size))

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=cfg.train.lr,
        total_steps=total_steps,
        final_div_factor=100,
        last_epoch=-1, pct_start=2/cfg.train.epochs
    )

    return optimizer, lr_scheduler


def prepare_CLIP(device=None):
    # Prepare CLIP
    clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
    clip_size = clip_sizes["ViT-L/14"]

    print("Using hidden layer CLIP space (Versatile Diffusion)")
    # if not args.norm_embs:
    #     print("WARNING: YOU WANT NORMED EMBEDDINGS FOR VERSATILE DIFFUSION!")
    clip_extractor = Clipper("ViT-L/14", device=device, hidden_state=True, norm_embs=False)

    out_dim_image = 257 * clip_size # 257*768 = 197376
    out_dim_text  = 77  * clip_size # 77*768  = 59136

    print("clip_extractor loaded.")
    print("out_dim_image:",out_dim_image)
    print("out_dim_text:", out_dim_text)

    return clip_extractor

import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor

import torch
import torch.nn as nn
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    # CLIPTextModel, # No longer used directly for encoder instance
    # CLIPVisionModel # No longer used directly for encoder instance
)

class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text and vision (from huggingface)"""

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
            version,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        # Load models with the specified dtype and ...WithProjection variants
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            version,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(device, dtype=dtype)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            version,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(device, dtype=dtype)
        # Add the image processor for preprocessing image inputs
        self.image_processor = CLIPImageProcessor.from_pretrained(
            version,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.device = device
        self.dtype = dtype # Store dtype for use, e.g., in image processing
        self.max_length = max_length

        # The hidden dimensionality of the text transformer's layers (pre-projection).
        self.text_model_hidden_size = self.text_encoder.config.hidden_size
        # The hidden dimensionality of the vision transformer's layers (pre-projection).
        self.image_model_hidden_size = self.image_encoder.config.hidden_size
        # The dimensionality of the final projected embeddings (shared space).
        self.projection_dim = self.text_encoder.config.projection_dim # or self.image_encoder.config.projection_dim
        
        if freeze:
            self.freeze()

    def freeze(self):
        self.text_encoder = self.text_encoder.eval()
        self.image_encoder = self.image_encoder.eval() # Also set image_encoder to eval mode
        
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text):
        """
        Encodes text using the CLIPTextModelWithProjection.
        Args:
            text (Union[str, List[str]]): The input text or list of texts to encode.
        Returns:
            dict: A dictionary containing:
                - "last_hidden_state": torch.Tensor of shape (batch_size, sequence_length, text_hidden_size)
                - "text_embeds": torch.Tensor of shape (batch_size, projection_dim) - The projected text embedding.
                - "attn_bias": torch.Tensor for attention masking.
        """
        if isinstance(text, str):
            text = [text] # Tokenizer expects a list of strings or a batch

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        text_outputs = self.text_encoder(
            input_ids=batch_encoding.input_ids,
            attention_mask=batch_encoding.attention_mask,
            output_hidden_states=True, # Ensure last_hidden_state is available if needed from base
            output_attentions=False
        )

        # Construct the attention bias
        attn_bias = batch_encoding["attention_mask"].to(text_outputs.last_hidden_state.dtype)
        attn_bias[attn_bias == 0] = -torch.inf
        attn_bias[attn_bias == 1] = 0.0

        return text_outputs.last_hidden_state
        # return {
        #     "last_hidden_state": text_outputs.last_hidden_state,
        #     "text_embeds": text_outputs.text_embeds, # Changed from pooler_output
        #     "attn_bias": attn_bias
        # }

    @torch.no_grad()
    def encode_image(self, image_input):
        """
        Encodes an image or a batch of images using the CLIPVisionModelWithProjection.
        Args:
            image_input: Input image(s). Can be PIL.Image, np.ndarray, torch.Tensor, List[PIL.Image], etc.,
                         as accepted by CLIPImageProcessor.
        Returns:
            dict: A dictionary containing:
                - "last_hidden_state": torch.Tensor (batch_size, num_patches + 1, image_hidden_size) # num_patches + 1 for [CLS] token
                - "image_embeds": torch.Tensor (batch_size, projection_dim) - The projected image embedding.
        """
        def versatile_normalize_embeddings(image_encoder, encoder_output):
            # embeds = encoder_output.last_hidden_state
            embeds = encoder_output.hidden_states
            embeds = [image_encoder.vision_model.post_layernorm(embeds[i]) for i in [12, 16, 20, 24]]
            # embeds = [image_encoder.vision_model.post_layernorm(embeds[i]) for i in [12,16,20,24,28,32,36,40,44,48]]
            # embeds = [image_encoder.vision_model.post_layernorm(embeds[i]) for i in [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]]
            # embeds = [image_encoder.vision_model.post_layernorm(embeds[i]) for i in range(15,24+1)]
            # embeds = image_encoder.vision_model.post_layernorm(embeds)
            embeds = [image_encoder.visual_projection(embed) for embed in embeds]
            return image_encoder.visual_projection(image_encoder.vision_model.post_layernorm(encoder_output.last_hidden_state)), embeds
        
        image_processed_inputs = self.image_processor(
            images=image_input,
            return_tensors="pt",
            do_rescale=False
        )
        
        pixel_values = image_processed_inputs.pixel_values.to(self.device, dtype=self.dtype)

        image_outputs = self.image_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True, # Ensure last_hidden_state is available if needed from base
            output_attentions=False
        )
        return versatile_normalize_embeddings(self.image_encoder, image_outputs)

        # return {
        #     "last_hidden_state": image_outputs.last_hidden_state,
        #     "image_embeds": image_outputs.image_embeds # Changed from pooler_output
        # }

    def forward(self, text_input=None, image_input=None):
        """
        Generic forward pass. Encodes text and/or image inputs.
        
        Args:
            text_input (Optional[Union[str, List[str]]]): Text to encode.
            image_input (Optional): Image(s) to encode.
            
        Returns:
            dict: A dictionary containing 'text_features' and/or 'image_features'.
                  If only one input is provided, the corresponding key will be present.
        
        Raises:
            ValueError: If neither text_input nor image_input is provided.
        """
        outputs = {}
        if text_input is not None:
            outputs["text_features"] = self.encode_text(text_input)
        
        if image_input is not None:
            outputs["image_features"] = self.encode_image(image_input)

        if not outputs:
            raise ValueError("Please provide at least text_input or image_input to forward().")
            
        return outputs

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x


def decode_latents(latents,vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity




def soft_clip_loss(preds, targs, temp=0.005, eps=1e-10):

    clip_clip = (targs @ targs.T)/temp + eps
    brain_clip = (preds @ targs.T)/temp + eps
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss


def clip_loss(image_features, text_features, temperature=1.0):
    """
    计算 CLIP 损失函数

    Args:
        image_features (torch.Tensor): 图像特征，形状为 (batch_size, feature_dim)
        text_features (torch.Tensor): 文本特征，形状为 (batch_size, feature_dim)
        temperature (float, optional): 温度参数，用于调节相似度计算的软化程度。默认值为 1.0。

    Returns:
        torch.Tensor: 损失值
    """

    # 计算 logits 矩阵
    logits_per_image = image_features @ text_features.T / temperature
    logits_per_text = text_features @ image_features.T / temperature

    # 创建 ground-truth labels
    labels = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=logits_per_image.device)

    # 计算交叉熵损失
    loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)

    return (loss_i + loss_t) / 2


def cross_modal_late_interaction(image_tokens, text_tokens):
    n1, n2 = image_tokens.size(1), text_tokens.size(1)  # Number of tokens per image and text
    
    # Compute token-wise similarities
    sim_matrix = torch.einsum('bik,bjk->bij', image_tokens, text_tokens)

    # Image-to-Text similarity
    max_sim_I = torch.max(sim_matrix, dim=2)[0]  # Max similarity for each image token across text tokens
    # sim_I = torch.mean(max_sim_I, dim=1)  # Average max similarity across image tokens
    loss_I2T = -torch.mean(F.log_softmax(max_sim_I, dim=1))

    # Text-to-Image similarity
    max_sim_T = torch.max(sim_matrix, dim=1)[0]  # Max similarity for each text token across image tokens
    # sim_T = torch.mean(max_sim_T, dim=1)  # Average max similarity across text tokens
    loss_T2I = -torch.mean(F.log_softmax(max_sim_T, dim=1))

    # Combine similarities for final cross-modal similarity score
    # cross_modal_similarity = (sim_I + sim_T) / 2
    cross_modal_similarity_loss = (loss_I2T + loss_T2I) / 2

    return cross_modal_similarity_loss

def combine_with_memory(
        pred_embedding_vision, pred_embedding_text, 
        clip_vision_train, clip_text_train,
        clip_vision_train_norm, clip_text_train_norm,
        clip_image_target, clip_text_target,
        clip_image_target_norm, clip_text_target_norm,
        retrival_from_memory_strength
        ):
    pred_embedding_text = pred_embedding_text.to(clip_text_train_norm.device)
    clip_image_target_norm = clip_image_target_norm.to(clip_text_train_norm.device)
    pred_embedding_vision = pred_embedding_vision.to(clip_text_train_norm.device)
    clip_text_target_norm = clip_text_target_norm.to(clip_text_train_norm.device)

    alpha = retrival_from_memory_strength
    
    pred_embedding_text_norm = nn.functional.normalize(pred_embedding_text.flatten(1), dim=-1)
    pred_embedding_vision_norm = nn.functional.normalize(pred_embedding_vision.flatten(1), dim=-1)
    
    # similarity_text = batchwise_cosine_similarity(pred_embedding_text_norm, clip_text_train_norm)
    # similarity_vision = batchwise_cosine_similarity(pred_embedding_vision_norm, clip_vision_train_norm)
    # similarity_text = clip_text_train_norm @ pred_embedding_text_norm.T
    # similarity_vision = clip_vision_train_norm @ pred_embedding_vision_norm.T
    
    # # Target with train data
    similarity_text_tar_with_train = batchwise_cosine_similarity(clip_text_target_norm, clip_text_train_norm)
    similarity_vision_tar_with_train = batchwise_cosine_similarity(clip_image_target_norm, clip_vision_train_norm) 
    topk_index_text_tar_with_train = torch.topk(similarity_text_tar_with_train.flatten(), 1).indices
    topk_index_vision_tar_with_train = torch.topk(similarity_vision_tar_with_train.flatten(), 1).indices

    # topk_index_text = torch.topk(similarity_text.flatten(), 1).indices
    # topk_index_vision = torch.topk(similarity_vision.flatten(), 1).indices
    # similarity_vision_by_text = pred_embedding_vision_norm @ (nn.functional.normalize(clip_vision_train[topk_index_text].flatten(1), dim=-1)).T
    # similarity_vision_by_text = batchwise_cosine_similarity(pred_embedding_vision_norm, (nn.functional.normalize(clip_vision_train[topk_index_text].flatten(1), dim=-1)))
    # print('\n Alpha' , alpha)
    # print('target 跟 train data 的最大相似度')
    # print('Top indices text tar retrival train data:',topk_index_text_tar_with_train,'Top similarity_text retrival train data:', torch.topk(similarity_text_tar_with_train.flatten(), 1).values)
    # print('Top indices vision tar retrival train data:',topk_index_vision_tar_with_train,'Top similarity_vision retrival train data', torch.topk(similarity_vision_tar_with_train.flatten(), 1).values)
    
    # print('预测在train data 中 top 10')
    # print('Top 10 index text:',torch.topk(similarity_text.flatten(), 10).indices,'Top 10 similarity_text:', torch.topk(similarity_text.flatten(), 10).values)  
    # print('Top 10 index vision:',torch.topk(similarity_vision.flatten(), 10).indices,'Top 10 similarity_vision', torch.topk(similarity_vision.flatten(), 10).values)

    # print('target 在train data 中 top 10')
    # print('Top 10 index text:',torch.topk(similarity_text_tar_with_train.flatten(), 10).indices,'Top 10 similarity_text:', torch.topk(similarity_text_tar_with_train.flatten(), 10).values)  
    # print('Top 10 index vision:',torch.topk(similarity_vision_tar_with_train.flatten(), 10).indices,'Top 10 similarity_vision', torch.topk(similarity_vision_tar_with_train.flatten(), 10).values)

    # print('预测与train data 中的最大相似度')
    # print('Top indices text:',topk_index_text,'Top similarity_text:', torch.topk(similarity_text.flatten(), 1).values)
    # print('Top indices vision:',topk_index_vision,'Top similarity_vision', torch.topk(similarity_vision.flatten(), 1).values)
    # print('Top similarity_vision_by_text',torch.topk(similarity_vision_by_text.flatten(), 1).values)

    

    # import ipdb;ipdb.set_trace()
    # combined_brain_clip_text_embeddings = (1-alpha) * clip_text_train[topk_index_text] + alpha * pred_embedding_text
    # combined_brain_clip_image_embeddings = (1-alpha) * clip_vision_train[topk_index_vision] + alpha * pred_embedding_vision
    # combined_brain_clip_image_embeddings_by_text = (1-alpha) * clip_vision_train[topk_index_text] + alpha * pred_embedding_vision

    combined_brain_clip_text_embeddings_using_target = (1-alpha) * clip_text_train[topk_index_text_tar_with_train] + alpha * pred_embedding_text
    combined_brain_clip_image_embeddings_using_target = (1-alpha) * clip_vision_train[topk_index_vision_tar_with_train] + alpha * pred_embedding_vision
    # combined_brain_clip_image_embeddings_by_text_using_target = (1-alpha) * clip_vision_train[topk_index_text_tar_with_train] + alpha * pred_embedding_vision


    # retrivaled_embedding_text_norm = nn.functional.normalize(clip_text_train[topk_index_text].flatten(1), dim=-1)
    # retrivaled_embedding_vision_norm = nn.functional.normalize(clip_vision_train[topk_index_vision].flatten(1), dim=-1)

    # similarity_text_retrival = batchwise_cosine_similarity(retrivaled_embedding_text_norm, clip_text_target_norm)
    # similarity_vision_retrival = batchwise_cosine_similarity(retrivaled_embedding_vision_norm, clip_image_target_norm)
    # print('检索到的跟tar之间的相似度')
    # print('Similarity_retrival_text_with_tar', similarity_text_retrival)
    # print('Similarity_retrival_vision_with_tar', similarity_vision_retrival)

    # combined_brain_clip_text_embeddings_norm = nn.functional.normalize(combined_brain_clip_text_embeddings.flatten(1), dim=-1)
    # combined_brain_clip_image_embeddings_norm = nn.functional.normalize(combined_brain_clip_image_embeddings.flatten(1), dim=-1)
    # combined_brain_clip_image_embeddings_by_text_norm = nn.functional.normalize(combined_brain_clip_image_embeddings_by_text.flatten(1), dim=-1)


    # combined_brain_clip_text_embeddings_norm_using_tar = nn.functional.normalize(combined_brain_clip_text_embeddings_using_target.flatten(1), dim=-1)
    # combined_brain_clip_image_embeddings_norm_using_tar = nn.functional.normalize(combined_brain_clip_image_embeddings_using_target.flatten(1), dim=-1)
    # combined_brain_clip_image_embeddings_by_text_norm_using_tar = nn.functional.normalize(combined_brain_clip_image_embeddings_by_text_using_target.flatten(1), dim=-1)

    # similarity_text_after_using_tar = batchwise_cosine_similarity(clip_text_target_norm , combined_brain_clip_text_embeddings_norm_using_tar)
    # similarity_vision_after_using_tar = batchwise_cosine_similarity(clip_image_target_norm , combined_brain_clip_image_embeddings_norm_using_tar)
    # similarity_vision_after_by_text_using_tar = batchwise_cosine_similarity(clip_image_target_norm , combined_brain_clip_image_embeddings_by_text_norm_using_tar)
    # print('使用 target 检索 进行合并后与 target 的相似度：')
    # print('Similarity_text_combined_with_tar_using_tar:', torch.topk(similarity_text_after_using_tar.flatten(), 1).values)
    # print('Similarity_vision_combined_with_combined_using_tar', torch.topk(similarity_vision_after_using_tar.flatten(), 1).values)
    # print('Similarity_vision_combined_with_combined_by_text_using_tar', torch.topk(similarity_vision_after_by_text_using_tar.flatten(), 1).values)

    # print('Similarity_text_combined_with_tar:', torch.topk(similarity_text_after.flatten(), 1).values)
    # print('Similarity_vision_combined_with_combined', torch.topk(similarity_vision_after.flatten(), 1).values)
    # print('Similarity_vision_combined_with_combined_by_text', torch.topk(similarity_vision_after_by_text.flatten(), 1).values)
    # similarity_text_before = pred_embedding_text_norm @ clip_text_target_norm.T
    # similarity_vision_before = pred_embedding_vision_norm @ clip_image_target_norm.T

    similarity_text_before = batchwise_cosine_similarity(pred_embedding_text_norm, clip_text_target_norm)
    similarity_vision_before = batchwise_cosine_similarity(pred_embedding_vision_norm, clip_image_target_norm)

    # similarity_text_after = clip_text_target_norm @ combined_brain_clip_text_embeddings_norm.T
    # similarity_vision_after = clip_image_target_norm @ combined_brain_clip_image_embeddings_norm.T
    # similarity_vision_after_by_text = clip_image_target_norm @ combined_brain_clip_image_embeddings_by_text_norm.T

    # similarity_text_after = batchwise_cosine_similarity(clip_text_target_norm , combined_brain_clip_text_embeddings_norm)
    # similarity_vision_after = batchwise_cosine_similarity(clip_image_target_norm , combined_brain_clip_image_embeddings_norm)
    # similarity_vision_after_by_text = batchwise_cosine_similarity(clip_image_target_norm , combined_brain_clip_image_embeddings_by_text_norm)
    # print('pred 跟 target 相似度')
    # print('Similarity_text_pred_with_tar:', torch.topk(similarity_text_before.flatten(), 1).values)
    # print('Similarity_vision_pred_with_tar', torch.topk(similarity_vision_before.flatten(), 1).values)
    # print('pred 自己检索合并后 跟 target 相似度')
    # print('Similarity_text_combined_with_tar:', torch.topk(similarity_text_after.flatten(), 1).values)
    # print('Similarity_vision_combined_with_tar', torch.topk(similarity_vision_after.flatten(), 1).values)
    # print('Similarity_vision_combined_with_tar_by_text', torch.topk(similarity_vision_after_by_text.flatten(), 1).values)

    # similarity_text_after_combine = batchwise_cosine_similarity(clip_text_train_norm , combined_brain_clip_text_embeddings_norm)
    # topk_index_text_after_combine = torch.topk(similarity_text_after_combine.flatten(), 1).indices

    # print('Top indices text after combine:',topk_index_text_after_combine,'Top similarity_text_after_combine:', torch.topk(similarity_text_after_combine.flatten(), 1).values)
    # combined_brain_clip_image_embeddings_by_text_after_combine = (1-alpha) * clip_vision_train[topk_index_text] + alpha * pred_embedding_vision
    # combined_brain_clip_image_embeddings_by_text_after_combine_norm = nn.functional.normalize(combined_brain_clip_image_embeddings_by_text_after_combine.flatten(1), dim=-1)
    # similarity_vision_after_by_text_after_combine = batchwise_cosine_similarity(clip_image_target_norm , combined_brain_clip_image_embeddings_by_text_after_combine_norm)
    # print('Similarity_vision_combined_with_combined_by_text_after_combine', torch.topk(similarity_vision_after_by_text_after_combine.flatten(), 1).values)


    # return combined_brain_clip_image_embeddings, combined_brain_clip_text_embeddings
    return combined_brain_clip_image_embeddings_using_target, combined_brain_clip_text_embeddings_using_target



@torch.no_grad()
def reconstruction(
    args,
    image, voxel, captions, 
    clip_vision_train, clip_text_train,
    clip_vision_train_norm, clip_text_train_norm,
    voxel2clip,
    clip_extractor,
    unet, vae, noise_scheduler,
    img_lowlevel = None,
    num_inference_steps = 50,
    recons_per_sample = 1,
    guidance_scale = 7.5,
    img2img_strength = .85,
    seed = 42,
    plotting=True,
    verbose=False,
    n_samples_save=1,
    device = None,
    mem_efficient = True,
    retrival_from_memory = False,
    retrival_from_memory_strength = 0.5,

):
    assert n_samples_save==1, "n_samples_save must = 1. Function must be called one image at a time"
    assert recons_per_sample>0, "recons_per_sample must > 0"
    
    brain_recons = None
    
    voxel=voxel[:n_samples_save]
    image=image[:n_samples_save]
    B = voxel.shape[0]

    clip_image_target = clip_extractor.embed_image(image)
    clip_text_target = clip_extractor.embed_text(captions)

    clip_image_target_norm = nn.functional.normalize(clip_image_target.flatten(1), dim=-1)
    clip_text_target_norm = nn.functional.normalize(clip_text_target.flatten(1), dim=-1)

    if mem_efficient:
        clip_extractor.to("cpu")
        unet.to("cpu")
        vae.to("cpu")
    else:
        clip_extractor.to(device)
        unet.to(device)
        vae.to(device)

    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if voxel2clip is not None:
        clip_results = voxel2clip(voxel)
        # clip_results = voxel2clip.backbone(ridge_out)
        if mem_efficient:
            voxel2clip.to('cpu')
        # brain_clip_text_embeddings = clip_extractor.embed_text(captions).float()
        brain_clip_image_embeddings, brain_clip_text_embeddings = clip_results[:2]
        if retrival_from_memory:
            brain_clip_image_embeddings, brain_clip_text_embeddings = combine_with_memory(
                brain_clip_image_embeddings, brain_clip_text_embeddings, 
                clip_vision_train, clip_text_train, 
                clip_vision_train_norm, clip_text_train_norm,
                clip_image_target, clip_text_target,
                clip_image_target_norm, clip_text_target_norm, retrival_from_memory_strength)
        # import ipdb;ipdb.set_trace()
        brain_clip_image_embeddings = brain_clip_image_embeddings.reshape(B,-1,768)
        brain_clip_text_embeddings  = brain_clip_text_embeddings.reshape(B,-1,768)

        brain_clip_image_embeddings = brain_clip_image_embeddings.repeat(recons_per_sample, 1, 1)
        brain_clip_text_embeddings  = brain_clip_text_embeddings.repeat(recons_per_sample, 1, 1)

    if recons_per_sample > 0:
        for samp in range(len(brain_clip_image_embeddings)):
            brain_clip_image_embeddings[samp] = brain_clip_image_embeddings[samp]/(brain_clip_image_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
            brain_clip_text_embeddings[samp]  = brain_clip_text_embeddings[samp]/(brain_clip_text_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        input_embedding = brain_clip_image_embeddings#.repeat(recons_per_sample, 1, 1)
        if verbose: print("input_embedding",input_embedding.shape)

        prompt_embeds = brain_clip_text_embeddings
        if verbose: print("prompt_embedding",prompt_embeds.shape)

        if do_classifier_free_guidance:
            input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
            prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)

        # 3. dual_prompt_embeddings
        input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)

        # 4. Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

        # 5b. Prepare latent variables
        batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
        shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        if img_lowlevel is not None: # use img_lowlevel for img2img initialization
            init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start:]
            latent_timestep = timesteps[:1].repeat(batch_size)
            
            if verbose: print("img_lowlevel", img_lowlevel.shape)
            img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
            if verbose: print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
            if mem_efficient:
                vae.to(device)
            init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
            init_latents = vae.config.scaling_factor * init_latents
            init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

            noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                                generator=generator, dtype=input_embedding.dtype)
            init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = init_latents
        else:
            timesteps = noise_scheduler.timesteps
            latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
                                  generator=generator, dtype=input_embedding.dtype)
            latents = latents * noise_scheduler.init_noise_sigma

        # 7. Denoising loop
        if mem_efficient:
            unet.to(device)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            if verbose: print("timesteps: {}, latent_model_input: {}, input_embedding: {}".format(i, latent_model_input.shape, input_embedding.shape))
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        if mem_efficient:
            unet.to("cpu")

        recons = decode_latents(latents.to(device),vae.to(device)).detach().cpu()

        brain_recons = recons.unsqueeze(0)

    if verbose: print("brain_recons",brain_recons.shape)
                    
    # pick best reconstruction out of several
    best_picks = np.zeros(n_samples_save).astype(np.int16)
    
    if mem_efficient:
        vae.to("cpu")
        unet.to("cpu")
        clip_extractor.to(device)

    # clip_image_target = clip_extractor.embed_image(image)
    # clip_image_target_norm = nn.functional.normalize(clip_image_target.flatten(1), dim=-1)
    sims=[]
    for im in range(recons_per_sample): 
        currecon = clip_extractor.embed_image(brain_recons[0,[im]].float()).to(clip_image_target_norm.device).to(clip_image_target_norm.dtype)
        currecon = nn.functional.normalize(currecon.view(len(currecon),-1),dim=-1)
        # import ipdb;ipdb.set_trace()
        cursim = batchwise_cosine_similarity(clip_image_target_norm,currecon)
        sims.append(cursim.item())
    if verbose: print(sims)
    best_picks[0] = int(np.nanargmax(sims))   
    if verbose: print(best_picks)
    if mem_efficient:
        clip_extractor.to("cpu")
        voxel2clip.to(device)
                    
    img2img_samples = 0 if img_lowlevel is None else 1
    num_xaxis_subplots = 1+img2img_samples+recons_per_sample
    if plotting:
        fig, ax = plt.subplots(n_samples_save, num_xaxis_subplots, 
                           figsize=(num_xaxis_subplots*5,6*n_samples_save),facecolor=(1, 1, 1))
    else:
        fig = None
        recon_img = None
    
    im = 0
    if plotting:
        ax[0].set_title(f"Original Image")
        ax[0].imshow(torch_to_Image(image[im]))
        if img2img_samples == 1:
            ax[1].set_title(f"Img2img ({img2img_strength})")
            ax[1].imshow(torch_to_Image(img_lowlevel[im].clamp(0,1)))
    for ii,i in enumerate(range(num_xaxis_subplots-recons_per_sample,num_xaxis_subplots)):
        recon = brain_recons[im][ii]
        if plotting:
            if ii == best_picks[im]:
                ax[i].set_title(f"Reconstruction",fontweight='bold')
                recon_img = recon
            else:
                ax[i].set_title(f"Recon {ii+1} from brain")
            ax[i].imshow(torch_to_Image(recon))
    if plotting:
        for i in range(num_xaxis_subplots):
            ax[i].axis('off')
    
    return fig, brain_recons, best_picks, recon_img

def read_responses_to_list(file_path):
    responses = []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_value = []
        
        for line in file:
            line = line.strip()
            if line.startswith("response_"):  # 新的 response 开始
                if current_value:  # 保存前一个 response
                    full_response = " ".join(current_value).strip()
                    responses.append(full_response.rstrip("</s>").strip())  # 去掉末尾的 </s>
                # 清空当前值并处理新 response
                parts = line.split(":", 1)
                current_value = [parts[1].strip()] if len(parts) > 1 else []
            else:  # 当前 response 的多行部分
                current_value.append(line)
        
        # 保存最后一个 response
        if current_value:
            full_response = " ".join(current_value).strip()
            responses.append(full_response.rstrip("</s>").strip())  # 去掉末尾的 </s>

    return responses
