# PLEASE MAKE SURE TO INCLUDE THE FOLLOWING BETWEEN THE START AND STOP COMMENTS:
#   1) Textual description of your system.
#   2) The code for your original system.
# PLEASE MAKE SURE NOT TO DELETE OR EDIT THE START AND STOP COMMENTS

# START COMMENT: Enter your system description in this cell.
from __future__ import annotations

import copy
import dataclasses
from datetime import datetime
from typing import List, Type, Set, Dict

import torch
from datasets import DatasetDict
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_WRITER = True
except:
    HAS_WRITER = False


from transformers import AutoTokenizer, AutoModel, BatchEncoding

import utils
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

DEFAULT_EXPERTS = [
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
]

EXPERT_SEP = -1


if HAS_WRITER:
    LOG = SummaryWriter(f"./runs/{datetime.now().strftime('%Y_%m_%d-%H_%M')}")



@dataclasses.dataclass
class ExpertConfig:
    n_classes: int = 3
    expert_hidden_activation: Type[nn.Module] = nn.ReLU
    arbiter_hidden_activation: Type[nn.Module] = nn.ReLU
    arbiter_hidden_dim: int = 128
    max_seq_length: int = 512


class Expert(nn.Module):
    def __init__(
        self,
        name: str,
        config: ExpertConfig,
        **kwargs,
    ):
        print(f"Loading expert: {name}")
        super().__init__(**kwargs)
        self.config = config
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name)
        self.hidden_dim = self.model.embeddings.word_embeddings.embedding_dim
        self.hidden_activation = self.config.expert_hidden_activation()

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.config.n_classes),
        )

    def encode_batch(self, examples: List[str]) -> torch.Tensor:
        toks = self.tok.batch_encode_plus(
            examples,
            add_special_tokens=True,
            max_length=self.config.max_seq_length,
            return_attention_mask=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return toks

    def decode_batch(self, examples: torch.Tensor) -> list[str]:
        dec = self.tok.batch_decode(examples)
        return dec

    def forward(
        self,
        indices: torch.LongTensor,
        mask: torch.LongTensor,
    ) -> torch.LongTensor:
        model_out = self.model(indices, attention_mask=mask, output_hidden_states=True)
        last_hidden = model_out.last_hidden_state
        preds = self.classifier(last_hidden[:, 0, :])
        return preds


class ExpertLayerWithArbiter(nn.Module):
    def __init__(
        self,
        experts: list[str] = None,
        config: ExpertConfig = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        experts = experts or DEFAULT_EXPERTS
        self.config = config or ExpertConfig()
        self.experts = nn.ModuleList([Expert(name, config) for name in experts])
        self.arbiter = nn.Sequential(
            nn.Linear(
                self.config.n_classes * len(self.experts),
                self.config.arbiter_hidden_dim,
            ),
            self.config.arbiter_hidden_activation(),
            nn.Linear(
                self.config.arbiter_hidden_dim,
                self.config.n_classes,
            ),
        )

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:

        sep_idxs = [0] + torch.where(inputs[0] == EXPERT_SEP)[0].tolist()
        assert len(sep_idxs) - 1 == len(self.experts), ValueError(
            f"Mismatch in number of separator indices ({len(sep_idxs)-1}) and number of experts ({len(self.experts)})!"
        )

        outs = []
        for expert_idx, expert in enumerate(self.experts):
            start = (
                sep_idxs[expert_idx] if expert_idx == 0 else sep_idxs[expert_idx] + 1
            )
            stop = sep_idxs[expert_idx + 1]
            expert_input = inputs[:, start:stop]
            expert_mask = masks[:, start:stop]
            expert_out = expert(expert_input.long(), expert_mask.long())
            outs.append(expert_out)

        # meow!
        cat = torch.cat(outs, dim=1)

        final = self.arbiter(cat)

        return final


class ExpertMixture(TorchShallowNeuralClassifier):

    model: ExpertLayerWithArbiter

    def __init__(
        self,
        experts: list[str] | None = None,
        config: ExpertConfig | None = None,
        **kwargs,
    ):
        self.experts = experts or DEFAULT_EXPERTS
        self.config = config or ExpertConfig()

        # set by .build_dataset()
        self.classes_: Set[str] | None = None
        self.n_classes_: int | None = None
        self.class2index: dict[str, int] | None = None

        super().__init__(**kwargs)

    def build_graph(self):
        return ExpertLayerWithArbiter(self.experts, self.config)

    def transform_for_experts(self, batch: Dict[str, list[str]]):
        encoded = []
        masks = []
        x_batch = batch["sentence"]
        for expert in self.model.experts:
            enc: BatchEncoding = expert.encode_batch(x_batch)
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
            batch_size = ids.shape[0]
            demarcator = (torch.zeros(batch_size) - 1).unsqueeze(1)

            # demarcate encodings and masks with -1
            # since -1 never appears in any encodings
            ids = torch.cat([ids, demarcator.clone()], dim=1)
            mask = torch.cat([mask, demarcator.clone()], dim=1)

            encoded.append(ids)
            masks.append(mask)

        y_batch = None
        if "gold_label" in batch:
            y_batch = batch["gold_label"]
            y_batch = [self.class2index[label] for label in y_batch]
            y_batch = torch.tensor(y_batch)

        return {
            "ids": torch.cat(encoded, dim=1),
            "masks": torch.cat(masks, dim=1),
            "labels": y_batch,
        }

    def build_dataset(self, X: DatasetDict, subset: str = "train", *args, **kwargs):
        x_subset = X[subset]

        if not self.classes_:
            y = x_subset["gold_label"]
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            self.class2index = dict(zip(self.classes_, range(self.n_classes_)))

        x_subset.set_transform(self.transform_for_experts)

        return x_subset

    def fit(self, dataset: DatasetDict, *args, **kwargs):
        # Set up parameters needed to use the model. This is a separate
        # function to support using pretrained models for prediction,
        # where it might not be desirable to call `fit`.
        self.initialize()

        # TODO: delete this!
        # self.device = "cpu" if self.device == "mps" else self.device

        # Make sure the model is where we want it:
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Dataset:
        train = self.build_dataset(dataset, "train")
        dataloader = self._build_dataloader(train, shuffle=self.shuffle_train)
        dev = []
        if self.early_stopping:
            dev = self.build_dataset(dataset, "validation")
            dev = self._build_dataloader(dev)

        self.model.train()
        self.optimizer.zero_grad()

        print("Begin fitting model")
        for iteration in range(1, self.max_iter + 1):

            epoch_error = 0.0

            for batch_num, batch in tqdm(
                enumerate(dataloader, start=1), total=len(dataloader)
            ):

                x_batch = batch["ids"].to(self.device), batch["masks"].to(self.device)
                y_batch = batch["labels"].to(self.device)

                batch_preds = self.model(*x_batch)

                err = self.loss(batch_preds, y_batch)

                if (
                    self.gradient_accumulation_steps > 1
                    and self.loss.reduction == "mean"
                ):
                    err /= self.gradient_accumulation_steps

                err.backward()
                if HAS_WRITER:
                    LOG.add_scalar("Loss/train", err.item(), batch_num)

                epoch_error += err.item()

                if (
                    batch_num % self.gradient_accumulation_steps == 0
                    or batch_num == len(dataloader)
                ):
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Stopping criteria:

            if self.early_stopping and dev:
                self._update_no_improvement_count_early_stopping(dev)
                if self.no_improvement_count > self.n_iter_no_change:
                    utils.progress_bar(
                        "Stopping after epoch {}. Validation score did "
                        "not improve by tol={} for more than {} epochs. "
                        "Final error is {}".format(
                            iteration, self.tol, self.n_iter_no_change, epoch_error
                        ),
                        verbose=self.display_progress,
                    )
                    break

            else:
                self._update_no_improvement_count_errors(epoch_error)
                if self.no_improvement_count > self.n_iter_no_change:
                    utils.progress_bar(
                        "Stopping after epoch {}. Training loss did "
                        "not improve more than tol={}. Final error "
                        "is {}.".format(iteration, self.tol, epoch_error),
                        verbose=self.display_progress,
                    )
                    break

            utils.progress_bar(
                "Finished epoch {} of {}; error is {}".format(
                    iteration, self.max_iter, epoch_error
                ),
                verbose=self.display_progress,
            )

        if self.early_stopping:
            self.model.load_state_dict(self.best_parameters)

        return self

    def _predict(self, dataloader: DataLoader):
        self.model.eval()
        preds = []
        y = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                x_batch = batch["ids"].to(self.device), batch["masks"].to(self.device)
                y_batch = batch["labels"].to(self.device)
                preds.append(self.model(*x_batch))
                y.append(y_batch)
        return torch.cat(preds, axis=0), torch.cat(y, axis=0)

    def _update_no_improvement_count_early_stopping(
        self, dev_loader: DataLoader, *args
    ):
        """
        Internal method used by `fit` to control early stopping.
        The method uses `self.score(*dev)` for scoring and updates
        `self.validation_scores`, `self.no_improvement_count`,
        `self.best_score`, `self.best_parameters` as appropriate.

        """

        preds, y = self._predict(dev_loader)
        y = y.cpu().detach().numpy()
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        score = utils.safe_macro_f1(y, preds)
        self.validation_scores.append(score)
        # If the score isn't at least `self.tol` better, increment:
        if score < (self.best_score + self.tol):
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        # If the current score is numerically better than all previous
        # scores, update the best parameters:
        if score > self.best_score:
            self.best_parameters = copy.deepcopy(self.model.state_dict())
            self.best_score = score
        self.model.train()


em = ExpertMixture(
    eta=0.00005,  # Low learning rate for effective fine-tuning.
    batch_size=8,  # Small batches to avoid memory overload.
    gradient_accumulation_steps=4,  # Increase the effective batch size to 32.
    early_stopping=True,  # Early-stopping
    n_iter_no_change=5,
)
dynasent_r1 = load_dataset("dynabench/dynasent", "dynabench.dynasent.r1.all")
em.fit(dynasent_r1)

# STOP COMMENT: Please do not remove this comment.
