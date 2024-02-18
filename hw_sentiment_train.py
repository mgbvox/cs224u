import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertModel, AutoModel

from custom_models.sentiment.bert_classifier import BertClassifierModule
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

import torch.nn as nn

from transformers import PreTrainedTokenizerFast, BatchEncoding


def get_batch_token_ids(
    batch: list[str], tokenizer: PreTrainedTokenizerFast
) -> BatchEncoding:
    """Map `batch` to a tensor of ids. The return
    value should meet the following specification:

    1. The max length should be 512.
    2. Examples longer than the max length should be truncated
    3. Examples should be padded to the max length for the batch.
    4. The special [CLS] should be added to the start and the special
       token [SEP] should be added to the end.
    5. The attention mask should be returned
    6. The return value of each component should be a tensor.

    Parameters
    ----------
    batch: list of str
    tokenizer: Hugging Face tokenizer

    Returns
    -------
    dict with at least "input_ids" and "attention_mask" as keys,
    each with Tensor values

    """
    ##### YOUR CODE HERE
    toks = tokenizer.batch_encode_plus(
        batch,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return toks


class BertClassifier(TorchShallowNeuralClassifier):
    def __init__(self, weights_name, *args, **kwargs):
        self.weights_name = weights_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_name)
        super().__init__(*args, **kwargs)
        self.params += ["weights_name"]

    def build_graph(self):
        return BertClassifierModule(
            self.n_classes_, self.hidden_activation, self.weights_name
        )

    def build_dataset(self, X, y=None):
        data = get_batch_token_ids(X, self.tokenizer)
        if y is None:
            dataset = torch.utils.data.TensorDataset(
                data["input_ids"], data["attention_mask"]
            )
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(
                data["input_ids"], data["attention_mask"], y
            )
        return dataset


def main():
    print("Starting HW Sentiment Training...")
    print("Build model")
    bert_finetune = BertClassifier(
        weights_name="prajjwal1/bert-mini",
        hidden_activation=nn.ReLU(),
        eta=0.00005,  # Low learning rate for effective fine-tuning.
        batch_size=8,  # Small batches to avoid memory overload.
        gradient_accumulation_steps=4,  # Increase the effective batch size to 32.
        early_stopping=True,  # Early-stopping
        n_iter_no_change=5,
    )  # params.

    print("Load dataset")
    dynasent_r1 = load_dataset("dynabench/dynasent", "dynabench.dynasent.r1.all")

    print("Fitting model")
    _ = bert_finetune.fit(
        dynasent_r1["train"]["sentence"], dynasent_r1["train"]["gold_label"]
    )


if __name__ == "__main__":
    main()
