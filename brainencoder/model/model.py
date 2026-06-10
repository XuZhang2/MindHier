import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerDecoder, TransformerDecoderLayer
class fMRIEncoder(nn.Module):
    def __init__(self, input_dim=15724, d_model=768, seq_len=100):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, seq_len*d_model)
        )

        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.d_model = d_model
        self.seq_len = seq_len

    def forward(self, x):
        x = self.proj(x)  # [batch, seq_len*d_model]
        x = x.view(x.size(0), self.seq_len, self.d_model)
        return x + self.pos_embed

class BranchDecoder(nn.Module):
    def __init__(self, num_tokens, d_model=768, nhead=8, num_layers=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_tokens, d_model))
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, d_model*4, activation='gelu'),
            num_layers
        )
        
    def forward(self, src):
        memory = src.permute(1, 0, 2)  # [seq_len, batch, d_model]
        tgt = self.query.unsqueeze(1).repeat(1, src.size(0), 1)  # [num_tokens, batch, d_model]
        out = self.decoder(tgt, memory)  # [num_tokens, batch, d_model]
        return out.permute(1, 0, 2)  # [batch, num_tokens, d_model]


class fMRI2CLIP(nn.Module):
    def __init__(self, 
                 input_dim=15724,
                 d_model=768,
                 fmri_seq_len=100,
                 image_seq_len=257,
                 text_seq_len=77, 
                 num_layers=4):
        super().__init__()
        
        self.fMRI_encoder = fMRIEncoder(input_dim, d_model, fmri_seq_len)
        self.image_decoder = BranchDecoder(image_seq_len, d_model, 8, num_layers)
        self.text_decoder = BranchDecoder(text_seq_len, d_model, 8, num_layers)
        
        self.image_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        fmri_features = self.fMRI_encoder(x)  # [batch, seq, d_model]
        
        image_emb = self.image_decoder(fmri_features)
        image_emb = self.image_proj(image_emb)
        
        text_emb = self.text_decoder(fmri_features) 
        text_emb = self.text_proj(text_emb)
        
        return image_emb, text_emb
    

