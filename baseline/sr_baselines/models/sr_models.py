import torch
import torch.nn as nn
import numpy as np
from math import sqrt


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SASRecModel(nn.Module):

    def __init__(self, item_num, hidden_size, max_seq_len, num_heads, num_blocks, dropout):
        super().__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(item_num + 1, hidden_size, padding_idx=0)
        self.pos_emb = PositionalEncoding(hidden_size, max_seq_len)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout, activation='gelu',
                batch_first=True
            ) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_emb = nn.Linear(hidden_size, item_num + 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, 0, 0.02)
        for layer in self.attention_layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)

    def forward(self, seq):
        mask = (seq == 0)
        x = self.item_emb(seq)
        x = self.pos_emb(x)
        x = self.dropout(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(seq.device)

        for layer in self.attention_layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=mask)

        x = self.layer_norm(x)
        logits = self.output_emb(x)
        return logits


class BERT4RecModel(nn.Module):

    def __init__(self, item_num, hidden_size, max_seq_len, num_heads, num_blocks, dropout, mask_ratio=0.15):
        super().__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.mask_token = item_num + 1

        self.item_emb = nn.Embedding(item_num + 2, hidden_size, padding_idx=0)
        self.pos_emb = PositionalEncoding(hidden_size, max_seq_len)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout, activation='gelu',
                batch_first=True
            ) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, item_num + 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, 0, 0.02)

    def forward(self, seq, masked_positions=None):
        mask = (seq == 0)
        x = self.item_emb(seq)
        x = self.pos_emb(x)
        x = self.dropout(x)

        attn_mask = None
        for layer in self.attention_layers:
            x = layer(x, src_key_padding_mask=mask)

        x = self.layer_norm(x)

        if masked_positions is not None:
            masked_output = x[masked_positions[:, 0], masked_positions[:, 1]]
            logits = self.output_proj(masked_output)
        else:
            logits = self.output_proj(x)

        return logits


class FDSAModel(nn.Module):

    def __init__(self, item_num, hidden_size, max_seq_len, num_heads, num_blocks, dropout, feature_dim):
        super().__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size

        self.item_emb = nn.Embedding(item_num + 1, hidden_size, padding_idx=0)
        self.feature_emb = nn.Linear(feature_dim, hidden_size)
        self.pos_emb = PositionalEncoding(hidden_size, max_seq_len)

        self.item_attention = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout, batch_first=True
            ) for _ in range(num_blocks)
        ])

        self.feature_attention = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout, batch_first=True
            ) for _ in range(num_blocks)
        ])

        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, item_num + 1)

    def forward(self, seq, features):
        mask = (seq == 0)
        x_item = self.item_emb(seq)
        x_item = self.pos_emb(x_item)
        x_item = self.dropout(x_item)

        x_feat = self.feature_emb(features)
        x_feat = self.dropout(x_feat)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(seq.device)

        for layer in self.item_attention:
            x_item = layer(x_item, src_mask=causal_mask, src_key_padding_mask=mask)

        for layer in self.feature_attention:
            x_feat = layer(x_feat, src_mask=causal_mask, src_key_padding_mask=mask)

        x = self.fusion_layer(torch.cat([x_item, x_feat], dim=-1))
        x = self.layer_norm(x)
        logits = self.output_proj(x)
        return logits


class S3RecModel(nn.Module):

    def __init__(self, item_num, hidden_size, max_seq_len, num_heads, num_blocks, dropout):
        super().__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size

        self.item_emb = nn.Embedding(item_num + 1, hidden_size, padding_idx=0)
        self.pos_emb = PositionalEncoding(hidden_size, max_seq_len)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout, batch_first=True
            ) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, item_num + 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, 0, 0.02)

    def forward(self, seq):
        mask = (seq == 0)
        x = self.item_emb(seq)
        x = self.pos_emb(x)
        x = self.dropout(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(seq.device)

        for layer in self.attention_layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=mask)

        x = self.layer_norm(x)
        logits = self.output_proj(x)
        return logits

    def compute_self_supervised_loss(self, seq, neg_seq):
        pos_logits = self.forward(seq)
        neg_logits = self.forward(neg_seq)
        pos_scores = pos_logits[:, -1, :]
        neg_scores = neg_logits[:, -1, :]
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return loss


class VQRecModel(nn.Module):

    def __init__(self, item_num, hidden_size, max_seq_len, num_heads, num_blocks, dropout,
                 codebook_size=256, num_codebooks=4, code_dim=64):
        super().__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.codebook = nn.Embedding(codebook_size * num_codebooks, code_dim)
        self.item_to_code_proj = nn.Linear(hidden_size, code_dim * num_codebooks)

        self.item_emb = nn.Embedding(item_num + 1, hidden_size, padding_idx=0)
        self.pos_emb = PositionalEncoding(hidden_size, max_seq_len)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout, batch_first=True
            ) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, item_num + 1)

    def forward(self, seq):
        mask = (seq == 0)
        x = self.item_emb(seq)
        x = self.pos_emb(x)
        x = self.dropout(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(seq.device)

        for layer in self.attention_layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=mask)

        x = self.layer_norm(x)
        logits = self.output_proj(x)
        return logits


class MISSRecModel(nn.Module):

    def __init__(self, item_num, hidden_size, max_seq_len, num_heads, num_blocks, dropout,
                 text_dim=4096, image_dim=768):
        super().__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size

        self.item_emb = nn.Embedding(item_num + 1, hidden_size, padding_idx=0)
        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.image_proj = nn.Linear(image_dim, hidden_size)
        self.modal_fusion = nn.Linear(hidden_size * 3, hidden_size)

        self.pos_emb = PositionalEncoding(hidden_size, max_seq_len)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout, batch_first=True
            ) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, item_num + 1)

    def forward(self, seq, text_features=None, image_features=None):
        mask = (seq == 0)
        x_id = self.item_emb(seq)

        if text_features is not None and image_features is not None:
            x_text = self.text_proj(text_features)
            x_image = self.image_proj(image_features)
            x = self.modal_fusion(torch.cat([x_id, x_text, x_image], dim=-1))
        else:
            x = x_id

        x = self.pos_emb(x)
        x = self.dropout(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(seq.device)

        for layer in self.attention_layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=mask)

        x = self.layer_norm(x)
        logits = self.output_proj(x)
        return logits
