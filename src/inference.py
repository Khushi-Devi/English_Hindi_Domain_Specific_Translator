import torch

from src.model import TransformerTranslator
from src.tokenizer import load_tokenizers


DEVICE = torch.device(
    "cpu"
)

MODEL_PATH = "models/legal_transformer.pt"


def load_translation_model():
    """
    Load trained Transformer model and tokenizers.
    """

    english_sp, hindi_sp = load_tokenizers()

    model = TransformerTranslator(
        src_vocab_size=english_sp.get_piece_size(),
        tgt_vocab_size=hindi_sp.get_piece_size()
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model, english_sp, hindi_sp


def translate(
    sentence,
    model,
    english_sp,
    hindi_sp,
    max_length=50
):
    """
    Translate an English sentence into Hindi.
    """

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
                (
                    target,
                    next_token.unsqueeze(0)
                ),
                dim=1
            )

            if next_token.item() == hindi_sp.eos_id():
                break

    tokens = target.squeeze().tolist()[1:]

    return hindi_sp.decode(tokens)