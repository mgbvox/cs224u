import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertModel, AutoModel

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


class BertClassifierModule(nn.Module):
    def __init__(
        self, n_classes, hidden_activation, weights_name="prajjwal1/bert-mini"
    ):
        """This module loads a Transformer based on  `weights_name`,
        puts it in train mode, add a dense layer with activation
        function give by `hidden_activation`, and puts a classifier
        layer on top of that as the final output. The output of
        the dense layer should have the same dimensionality as the
        model input.

        Parameters
        ----------
        n_classes : int
            Number of classes for the output layer
        hidden_activation : torch activation function
            e.g., nn.Tanh()
        weights_name : str
            Name of pretrained model to load from Hugging Face

        """
        super().__init__()
        self.n_classes = n_classes
        self.weights_name = weights_name
        self.bert: BertModel = AutoModel.from_pretrained(self.weights_name)
        self.bert.train()
        self.hidden_activation = hidden_activation
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        # Add the new parameters here using `nn.Sequential`.
        # We can define this layer as
        #
        #  h = f(cW1 + b_h)
        #  y = hW2 + b_y
        #
        # where c is the final hidden state above the [CLS] token,
        # W1 has dimensionality (self.hidden_dim, self.hidden_dim),
        # W2 has dimensionality (self.hidden_dim, self.n_classes),
        # f is the hidden activation, and we rely on the PyTorch loss
        # function to add apply a softmax to y.
        ##### YOUR CODE HERE
        self.classifier_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.n_classes),
        )

    def forward(self, indices, mask):
        """Process `indices` with `mask` by feeding these arguments
        to `self.bert` and then feeding the initial hidden state
        in `last_hidden_state` to `self.classifier_layer`

        Parameters
        ----------
        indices : tensor.LongTensor of shape (n_batch, k)
            Indices into the `self.bert` embedding layer. `n_batch` is
            the number of examples and `k` is the sequence length for
            this batch
        mask : tensor.LongTensor of shape (n_batch, d)
            Binary vector indicating which values should be masked.
            `n_batch` is the number of examples and `k` is the
            sequence length for this batch

        Returns
        -------
        tensor.FloatTensor
            Predicted values, shape `(n_batch, self.n_classes)`

        """
        ##### YOUR CODE HERE

        bert_out = self.bert(indices, attention_mask=mask, output_hidden_states=True)
        # (batch_size, seq_len, hidden_dim)
        last_hidden = bert_out.last_hidden_state
        # feed the initial hidden state of the final hidden layer into self.classifier_layer
        preds = self.classifier_layer(last_hidden[:, 0, :])
        return preds


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

if __name__ == '__main__':
    main()