# PLEASE MAKE SURE TO INCLUDE THE FOLLOWING BETWEEN THE START AND STOP COMMENTS:
#   1) Textual description of your system.
#   2) The code for your original system.
# PLEASE MAKE SURE NOT TO DELETE OR EDIT THE START AND STOP COMMENTS

# START COMMENT: Enter your system description in this cell.
from __future__ import annotations

import copy
import dataclasses
from datetime import datetime, time
from pathlib import Path
from typing import List, Type, Set, Dict, Tuple

import numpy as np
import torch
from datasets import DatasetDict, Dataset
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wget
import pandas as pd

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

START_DATETIME = datetime.now().strftime("%Y_%m_%d-%H_%M")

if HAS_WRITER:
    LOG = SummaryWriter(f"./runs/{START_DATETIME}")

CHECKPOINT_ROOT = Path(f"./checkpoints/{START_DATETIME}")
CHECKPOINT_ROOT.mkdir(exist_ok=True, parents=True)

BAKEOFF_ROOT = Path(f"./bakeoff/{START_DATETIME}")
BAKEOFF_ROOT.mkdir(exist_ok=True, parents=True)


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
            f"Mismatch in number of separator indices ({len(sep_idxs) - 1}) and number of experts ({len(self.experts)})!"
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
        self.model_class = ExpertLayerWithArbiter

        self.experts = experts or DEFAULT_EXPERTS
        self.config = config or ExpertConfig()

        # set by .build_dataset()
        self.classes_: list[str] | None = None
        self.n_classes_: int | None = None
        self.class2index: dict[str, int] | None = None

        # set by .setup_bakeoff()
        self.bakeoff_df: pd.DataFrame | None = None
        self.bakeoff_loader: DataLoader | None = None

        super().__init__(**kwargs)

    def build_graph(self) -> ExpertLayerWithArbiter:
        return self.model_class(self.experts, self.config)

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

        to_return = {
            "ids": torch.cat(encoded, dim=1),
            "masks": torch.cat(masks, dim=1),
            "labels": y_batch,
        }
        if y_batch:
            _ = to_return.pop("labels")
        return to_return

    def build_dataset_from_csv(self, csv: str):
        ds = Dataset.from_csv(csv)
        ds.set_transform(self.transform_for_experts)
        return ds

    def build_dataset(self, X: DatasetDict, subset: str = "train", *args, **kwargs):
        x_subset = X[subset]

        if not self.classes_:
            y = x_subset["gold_label"]
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            self.class2index = dict(zip(self.classes_, range(self.n_classes_)))

        x_subset.set_transform(self.transform_for_experts)

        return x_subset

    def initialize(self, checkpoint_path: Path | None = None):
        """
        Adapted to allow loading from checkpoints
        """
        self.model = self.build_graph()
        # This device move has to happen before the optimizer is built:
        # https://pytorch.org/docs/master/optim.html#constructing-it
        self.optimizer = self.build_optimizer()
        if checkpoint_path:
            print(f"Initializing from checkpoint {checkpoint_path}")
        data = torch.load(checkpoint_path) if checkpoint_path else dict()
        self.best_parameters = data.get("model_state_dict")
        if self.best_parameters:
            self.model.load_state_dict(self.best_parameters)
        optim_state = data.get("optimizer_state_dict")
        if optim_state:
            self.optimizer.load_state_dict(optim_state)
        self.best_error = data.get("best_error", np.inf)
        self.best_score = data.get("best_score", -np.inf)

        self.errors = []
        self.validation_scores = []
        self.no_improvement_count = 0

        print(
            f"Model initialized; best_err: {self.best_error}; best_score: {self.best_score}"
        )

    def setup_bakeoff(self):
        # for use in submitting bakeoff
        if not os.path.exists(
            os.path.join("data", "sentiment", "cs224u-sentiment-test-unlabeled.csv")
        ):
            os.makedirs(os.path.join("data", "sentiment"), exist_ok=True)
            wget.download(
                "https://web.stanford.edu/class/cs224u/data/cs224u-sentiment-test-unlabeled.csv",
                out="data/sentiment/",
            )

        csv = os.path.join("data", "sentiment", "cs224u-sentiment-test-unlabeled.csv")
        self.bakeoff_df = pd.read_csv(csv)
        bakeoff_ds = self.build_dataset_from_csv(csv)
        self.bakeoff_loader = self._build_dataloader(bakeoff_ds, shuffle=False)

    def fit(
        self,
        dataset: DatasetDict,
        checkpoint_path: Path | str | None = None,
        *args,
        **kwargs,
    ):
        # Set up parameters needed to use the model. This is a separate
        # function to support using pretrained models for prediction,
        # where it might not be desirable to call `fit`.
        self.initialize(checkpoint_path)
        # mps running out of memory (big models); do this on cpu for now
        self.device = "cpu" if str(self.device) == "mps" else self.device
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

        self.setup_bakeoff()
        # do a test prediction of the benchmark data to ensure we're good to infer later
        self.bakeoff(test=True)

        self.model.train()
        self.optimizer.zero_grad()

        print("Initial model checkpoint (pre-training)!")
        self.checkpoint()

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

    def _predict_batch(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        did_toggle = False
        if self.model.training:
            self.model.eval()

        if torch.is_grad_enabled():
            # disable grad for preds!
            with torch.no_grad():
                x_batch = batch["ids"].to(self.device), batch["masks"].to(self.device)
                out_batch = self.model(*x_batch)
        else:
            # grad already disabled, no need to call again
            x_batch = batch["ids"].to(self.device), batch["masks"].to(self.device)
            out_batch = self.model(*x_batch)

        if not self.model.training and did_toggle:
            self.model.train()

        return out_batch

    def _predict(
        self, dataloader: DataLoader, stop_after: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        self.model.eval()
        preds: list[torch.Tensor] = []
        y: list[torch.Tensor] = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                assert isinstance(batch, dict)
                out_batch = self._predict_batch(batch)
                if "labels" in batch.keys():
                    y_batch = batch["labels"]
                    y.append(torch.tensor(y_batch))
                preds.append(torch.tensor(out_batch))
                if stop_after is not None and idx > stop_after:
                    break

        self.model.train()

        return torch.cat(preds, dim=0), (torch.cat(y, dim=0) if y else None)

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
            # write weights out to storage for loading later!
            self.checkpoint()

    def checkpoint(self):
        checkpoint_model(net=self)
        # update bakeoff submission
        self.bakeoff()

    def bakeoff(self, test=False):
        print(f"Bakeoff mode; test = {test}")
        preds, _ = self._predict(self.bakeoff_loader, stop_after=10 if test else None)
        probs = torch.softmax(torch.tensor(preds), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        labels = [self.classes_[i] for i in preds]
        if len(labels) != len(self.bakeoff_df):
            labels += ["None" for _ in range(len(self.bakeoff_df) - len(labels))]
        self.bakeoff_df["prediction"] = labels

        now = datetime.now().strftime("DATE_%Y_%m_%d-TIME_%H_%M")
        end = (
            "TEST" if test else f"{now}-SCORE_{self.best_score}-LOSS_{self.best_error}"
        )
        out = BAKEOFF_ROOT / f"bakeoff_submission-{end}.csv"
        self.bakeoff_df.to_csv(out)
        print(f"Wrote bakeoff submission to {out}")


def checkpoint_model(net: ExpertMixture):
    name = net.__class__.__name__
    now = datetime.now().strftime("DATE_%Y_%m_%d-TIME_%H_%M")
    if not hasattr(net, "model"):
        raise ValueError(f"Model not yet initialized for {name}!")

    score = net.best_score if net.best_score else 0
    loss = min(net.best_error, 99999) if net.best_error else 99999
    path = CHECKPOINT_ROOT / f"chkpt-{name}-{now}-SCORE_{score}-LOSS_{loss}.ckpt"
    print(f"Checkpointing model for:\n\t{name}\nto:\n\t{path}")
    torch.save(
        {
            "model_state_dict": net.model.state_dict(),
            "optimizer_state_dict": net.optimizer.state_dict(),
            "best_error": loss,
            "best_score": score,
        },
        path,
    )


def main():
    em = ExpertMixture(
        eta=0.00005,  # Low learning rate for effective fine-tuning.
        batch_size=8,  # Small batches to avoid memory overload.
        gradient_accumulation_steps=4,  # Increase the effective batch size to 32.
        early_stopping=True,  # Early-stopping
        n_iter_no_change=5,
    )
    dynasent_r1 = load_dataset("dynabench/dynasent", "dynabench.dynasent.r1.all")
    em.fit(dynasent_r1)


if __name__ == "__main__":
    main()
