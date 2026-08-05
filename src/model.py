import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4
    ):
        super().__init__()

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model
        )

        self.position = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            d_model,
            tgt_vocab_size
        )

    def forward(self, src, tgt):

        src = self.position(
            self.src_embedding(src)
        )

        tgt = self.position(
            self.tgt_embedding(tgt)
        )

        output = self.transformer(
            src,
            tgt
        )

        output = self.fc(output)

        return output