import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm
import math

DEVICE = torch.device(
    "cpu"
)
st.set_page_config(
    page_title="English Hindi Legal Translator",
    page_icon="⚖️"
)
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
            *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(
            position * div_term
        )
        pe[:, 1::2] = torch.cos(
            position * div_term
        )
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
        d_model=64,
        nhead=4,
        num_layers=1
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
        self.position = PositionalEncoding(
            d_model
        )
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
   
@st.cache_resource
def load_model():

    english_tokenizer = spm.SentencePieceProcessor()
    english_tokenizer.load(
        "english_tokenizer.model"
    )

    hindi_tokenizer = spm.SentencePieceProcessor()
    hindi_tokenizer.load(
        "hindi_tokenizer.model"
    )

    model = TransformerTranslator(
        src_vocab_size=english_tokenizer.get_piece_size(),
        tgt_vocab_size=hindi_tokenizer.get_piece_size()
    )

    model.load_state_dict(
        torch.load(
            "models/legal_transformer.pt",
            map_location=DEVICE
        )
    )
    model.to(DEVICE)
    model.eval()
    return model, english_tokenizer, hindi_tokenizer

model, english_sp, hindi_sp = load_model()

def translate(sentence, max_length=50):

    source_tokens = english_sp.encode(
        sentence,
        out_type=int
    )

    source = torch.tensor(
        source_tokens
    ).unsqueeze(0).to(DEVICE)

    target = torch.tensor(
        [[hindi_sp.bos_id()]]
    ).to(DEVICE)

    with torch.no_grad():

        for _ in range(max_length):

            output = model(
                source,
                target
            )

            logits = output[:, -1, :]
            logits[:, 0] = -float("inf")
            next_token = logits.argmax(
                dim=-1
            )
            target = torch.cat(
                [
                    target,
                    next_token.unsqueeze(0)
                ],
                dim=1
            )
            if next_token.item() == hindi_sp.eos_id():
                break

    tokens = target.squeeze().tolist()[1:]
    return hindi_sp.decode(tokens)

st.title(
    "⚖️ English-Hindi Legal Translator"
)


st.write(
    """
    Transformer-based English to Hindi
    legal translation system.
    """
)
text = st.text_area(
    "Enter English legal text"
)
if st.button("Translate"):

    if text.strip():

        result = translate(text)

        st.subheader(
            "Hindi Translation"
        )

        st.success(result)

    else:

        st.warning(
            "Please enter text."
        )
st.sidebar.title(
    "Project Details"
)
st.sidebar.write(
    """
    Model:
    Transformer Encoder-Decoder
    Framework:
    PyTorch
    Domain:
    Legal Translation
    """
)