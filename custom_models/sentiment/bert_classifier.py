import torch
from torch import nn
from transformers import BertModel, AutoModel


class BertClassifierModule(nn.Module):
    def __init__(
        self,
        n_classes: int,
        hidden_activation: nn.Module,
        weights_name: str = "prajjwal1/bert-mini",
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

    def forward(self, indices: torch.LongTensor, mask: torch.LongTensor):
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
        bert_out = self.bert(indices, attention_mask=mask, output_hidden_states=True)
        # (batch_size, seq_len, hidden_dim)
        last_hidden = bert_out.last_hidden_state
        # feed the initial hidden state of the final hidden layer into self.classifier_layer
        preds = self.classifier_layer(last_hidden[:, 0, :])
        return preds
