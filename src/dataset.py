import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(
        self,
        dataframe,
        english_tokenizer,
        hindi_tokenizer,
        max_length=50
    ):
        self.dataframe = dataframe
        self.english_tokenizer = english_tokenizer
        self.hindi_tokenizer = hindi_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        english_sentence = self.dataframe.iloc[index]["english"]
        hindi_sentence = self.dataframe.iloc[index]["hindi"]

        source = self.english_tokenizer.encode(
            english_sentence,
            out_type=int
        )

        target = self.hindi_tokenizer.encode(
            hindi_sentence,
            out_type=int
        )

        source = source[:self.max_length]
        target = target[:self.max_length - 2]

        target = (
            [self.hindi_tokenizer.bos_id()]
            + target
            + [self.hindi_tokenizer.eos_id()]
        )

        source += [0] * (
            self.max_length - len(source)
        )

        target += [0] * (
            self.max_length - len(target)
        )

        return (
            torch.tensor(source),
            torch.tensor(target)
        )