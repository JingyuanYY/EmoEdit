import torch
import torch.nn as nn
from transformers import (
    Blip2QFormerConfig,
    Blip2QFormerModel,
)
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F



class CombinedModel(nn.Module):
    def __init__(self, num_query_tokens, hidden_size=768, ln=None):
        super(CombinedModel, self).__init__()
        qformer_config = Blip2QFormerConfig()
        qformer_config.encoder_hidden_size = 768
        qformer_config.attention_probs_dropout_prob = 0.0
        self.model = Blip2QFormerModel(qformer_config)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        self.layer_norm = ln if ln is not None else nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, text_embeds, img_embeds):
        # Expand query tokens to match batch size
        batch_size = img_embeds.size(0)
        expand_query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Concatenate query tokens and text embeddings
        query = torch.cat([expand_query_tokens, text_embeds], dim=1)

        # Create image attention mask
        image_attention_mask = torch.ones(img_embeds.size()[:-1], dtype=img_embeds.dtype, device=img_embeds.device)

        # Forward pass through the base model
        query_outputs = self.model(
            query_embeds=query,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=image_attention_mask,
        )[0]

        # Apply layer normalization
        normal_output = self.layer_norm(query_outputs)

        return normal_output









