import sentencepiece as spm


def load_tokenizers():
    """
    Load the trained English and Hindi SentencePiece tokenizers.
    """

    english_tokenizer = spm.SentencePieceProcessor()
    english_tokenizer.load(
        "models/english_tokenizer.model"
    )

    hindi_tokenizer = spm.SentencePieceProcessor()
    hindi_tokenizer.load(
        "models/hindi_tokenizer.model"
    )

    return english_tokenizer, hindi_tokenizer


def train_tokenizer(
    input_file,
    model_prefix,
    vocab_size=4000
):
    """
    Train a SentencePiece tokenizer.
    """

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe"
    )

def create_tokenizer_corpus(
    train_df,
    english_output,
    hindi_output
):
    """
    Create text files used for SentencePiece training.
    """

    with open(
        english_output,
        "w",
        encoding="utf-8"
    ) as file:

        for sentence in train_df["english"]:
            file.write(sentence + "\n")

    with open(
        hindi_output,
        "w",
        encoding="utf-8"
    ) as file:

        for sentence in train_df["hindi"]:
            file.write(sentence + "\n")

    print("Tokenizer corpus created successfully.")