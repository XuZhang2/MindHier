import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from utils.constants import SUBJECT_NUM_VOXELS

class FMRIEncoder(nn.Module):
    """
    Encodes fMRI data into a sequence of embeddings.
    """
    def __init__(
        self,
        input_dim=SUBJECT_NUM_VOXELS[1],
        d_model=768,
        seq_len=100,
        multi_subj=False,
        subject_num_voxels=None,
    ):
        """
        Initializes the FMRIEncoder.

        Args:
            input_dim (int): Dimensionality of the input fMRI data.
            d_model (int): Dimensionality of the model's hidden states.
            seq_len (int): Length of the output sequence.
        """
        super().__init__()
        self.multi_subj = multi_subj
        self.input_sizes = list((subject_num_voxels or SUBJECT_NUM_VOXELS).values())
        if self.multi_subj:
            self.ridge = nn.ModuleList([
                nn.Linear(input_size, 4096) for input_size in self.input_sizes
            ])
            self.proj = nn.Sequential(
                nn.GELU(),
                nn.Linear(4096, 2048),
                nn.GELU(),
                nn.Linear(2048, seq_len * d_model)
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, 4096),
                nn.GELU(),
                nn.Linear(4096, 2048),
                nn.GELU(),
                nn.Linear(2048, seq_len * d_model)
            )
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.d_model = d_model
        self.seq_len = seq_len

    def forward(self, x):
        """
        Forward pass of the FMRIEncoder.

        Args:
            x (torch.Tensor): Input fMRI data with shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Encoded fMRI features with shape [batch_size, seq_len, d_model].
        """
        if self.multi_subj:
            out = []
            for sample in x:
                input_dim = sample.shape[-1]
                if input_dim not in self.input_sizes:
                    raise ValueError(
                        f"Unknown voxel dimension {input_dim}. "
                        "Update SUBJECT_NUM_VOXELS if you use a custom NSD mask."
                    )
                ridge_idx = self.input_sizes.index(input_dim)
                out.append(self.ridge[ridge_idx](sample))
            x = torch.stack(out)
        elif isinstance(x, (list, tuple)):
            x = torch.stack(x)

        x = self.proj(x)  # [batch_size, seq_len * d_model]
        # Reshape to [batch_size, seq_len, d_model]
        x = x.view(x.size(0), self.seq_len, self.d_model)
        # Add positional embeddings
        return x + self.pos_embed

class IntermediateTransformerDecoder(TransformerDecoder):
    """
    Wrapper class around PyTorch's TransformerDecoder to extract intermediate
    features from each decoder layer, while maintaining the original parameter structure.
    """
    def __init__(self, decoder_layer, num_layers, norm=None):
        """
        Initializes the IntermediateTransformerDecoder.

        Args:
            decoder_layer (nn.TransformerDecoderLayer): An instance of TransformerDecoderLayer.
            num_layers (int): The number of sub-decoder-layers in the decoder.
            norm (nn.Module, optional): An optional layer normalization component.
        """
        super().__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the IntermediateTransformerDecoder.

        Args:
            tgt (torch.Tensor): The sequence to the decoder (target).
                                Shape: [target_seq_len, batch_size, d_model].
            memory (torch.Tensor): The sequence from the last layer of the encoder (memory).
                                   Shape: [source_seq_len, batch_size, d_model].
            tgt_mask (torch.Tensor, optional): The additive mask for the target sequence.
            memory_mask (torch.Tensor, optional): The additive mask for the memory sequence.
            tgt_key_padding_mask (torch.Tensor, optional): The byte mask for the target sequence.
            memory_key_padding_mask (torch.Tensor, optional): The byte mask for the memory sequence.

        Returns:
            tuple:
                - torch.Tensor: The final output of the transformer decoder.
                                Shape: [target_seq_len, batch_size, d_model].
                - list[torch.Tensor]: A list of intermediate outputs from each decoder layer.
                                      Each tensor has shape [target_seq_len, batch_size, d_model].
        """
        intermediate = []
        output = tgt

        # Iterate through each decoder layer
        for mod in self.layers:
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            intermediate.append(output)

        # Apply normalization if it exists
        if self.norm is not None:
            output = self.norm(output)
            intermediate[-1] = output  # Update the last intermediate output with the normalized version

        return output, intermediate  # Return both final output and intermediate features

class BranchDecoder(nn.Module):
    """
    A decoder module that uses a TransformerDecoder to process source features
    with learned query embeddings. It returns both the final output and
    intermediate layer outputs.
    """
    def __init__(self, num_tokens, d_model=768, nhead=8, num_layers=4, dim_feedforward=3072, dropout=0.1, activation='gelu'):
        """
        Initializes the BranchDecoder.

        Args:
            num_tokens (int): The number of query tokens (determines output sequence length).
            d_model (int): Dimensionality of the model's hidden states.
            nhead (int): Number of attention heads in the TransformerDecoderLayer.
            num_layers (int): Number of layers in the TransformerDecoder.
            dim_feedforward (int): Dimension of the feedforward network model in TransformerDecoderLayer.
            dropout (float): Dropout value in TransformerDecoderLayer.
            activation (str): Activation function in TransformerDecoderLayer ('relu' or 'gelu').
        """
        super().__init__()
        # Learnable query parameters for the decoder
        self.query = nn.Parameter(torch.randn(num_tokens, d_model))

        # Transformer decoder layer
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first=False
        ) # TransformerDecoder expects (seq, batch, feature)

        # IntermediateTransformerDecoder to get outputs from all layers
        self.decoder = IntermediateTransformerDecoder(
            decoder_layer,
            num_layers
        )
        self.d_model = d_model
        self.num_tokens = num_tokens

    def forward(self, src):
        """
        Forward pass of the BranchDecoder.

        Args:
            src (torch.Tensor): Source tensor from the encoder.
                                Shape: [batch_size, src_seq_len, d_model].

        Returns:
            dict: A dictionary containing:
                - 'final' (torch.Tensor): The final output of the decoder.
                                          Shape: [batch_size, num_tokens, d_model].
                - 'intermediates' (list[torch.Tensor]): List of intermediate outputs
                                                        from each decoder layer. Each tensor
                                                        has shape [batch_size, num_tokens, d_model].
        """
        # Permute src to [src_seq_len, batch_size, d_model] for TransformerDecoder
        memory = src.permute(1, 0, 2)

        # Prepare target tensor (queries)
        # Unsqueeze to add batch dimension and repeat for each item in the batch
        # Shape: [num_tokens, batch_size, d_model]
        tgt = self.query.unsqueeze(1).repeat(1, src.size(0), 1)

        # Pass through the decoder
        final_output, intermediates = self.decoder(tgt, memory)

        # Permute outputs back to [batch_size, num_tokens, d_model]
        final_output_permuted = final_output.permute(1, 0, 2)
        intermediates_permuted = [x.permute(1, 0, 2) for x in intermediates]

        return {
            'final': final_output_permuted,
            'intermediates': intermediates_permuted
        }

class FMRI2CLIP(nn.Module):
    """
    Main model to transform fMRI signals into CLIP-like image and text embeddings.
    """
    def __init__(self,
                 input_dim=SUBJECT_NUM_VOXELS[1],
                 d_model=768,
                 fmri_seq_len=100,
                 image_seq_len=257, # Standard ViT-B/16 patch + CLS token
                 text_seq_len=77,   # Standard CLIP text model sequence length
                 nhead=8,
                 num_layers=4,
                 multi_subj=False,
                ):
        """
        Initializes the FMRI2CLIP model.

        Args:
            input_dim (int): Dimensionality of the input fMRI data.
            d_model (int): Dimensionality of the model's hidden states.
            fmri_seq_len (int): Sequence length of the fMRI encoder output.
            image_seq_len (int): Sequence length for the image embedding branch.
            text_seq_len (int): Sequence length for the text embedding branch.
            nhead (int): Number of attention heads for branch decoders.
            num_layers (int): Number of layers for branch decoders.
            dim_feedforward (int): Feedforward dimension for branch decoders.
        """
        super().__init__()

        # fMRI Encoder
        self.fMRI_encoder = FMRIEncoder(input_dim, d_model, fmri_seq_len, multi_subj=multi_subj)

        # Image Decoder Branch
        self.image_decoder = BranchDecoder(image_seq_len, d_model, nhead, num_layers, d_model*4)
        # Text Decoder Branch
        self.text_decoder = BranchDecoder(text_seq_len, d_model, nhead, num_layers, d_model*4)

        # Projection layers to map decoder outputs to final embedding space (optional, could be identity)
        # These layers expect a tensor of shape [batch_size, seq_len, d_model]
        self.image_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Forward pass of the FMRI2CLIP model.

        Args:
            x (torch.Tensor): Input fMRI data with shape [batch_size, input_dim].

        Returns:
            tuple:
                - torch.Tensor: Projected image embeddings.
                                Shape: [batch_size, image_seq_len, d_model].
                - torch.Tensor: Projected text embeddings.
                                Shape: [batch_size, text_seq_len, d_model].
        """
        # Encode fMRI data
        # Output shape: [batch_size, fmri_seq_len, d_model]
        fmri_features = self.fMRI_encoder(x)

        # Decode into image embedding space
        # image_decoder_output is a dictionary: {'final': tensor, 'intermediates': [tensor, ...]}
        image_decoder_output = self.image_decoder(fmri_features)
        # We need the 'final' tensor for projection
        # Shape: [batch_size, image_seq_len, d_model]
        image_emb_final = image_decoder_output['final']
        projected_image_emb = self.image_proj(image_emb_final)
        image_emb_all = image_decoder_output['intermediates']
  

        # Decode into text embedding space
        # text_decoder_output is a dictionary: {'final': tensor, 'intermediates': [tensor, ...]}
        text_decoder_output = self.text_decoder(fmri_features)
        # We need the 'final' tensor for projection
        # Shape: [batch_size, text_seq_len, d_model]
        text_emb_final = text_decoder_output['final']
        projected_text_emb = self.text_proj(text_emb_final)
        return projected_image_emb, projected_text_emb, image_emb_all
