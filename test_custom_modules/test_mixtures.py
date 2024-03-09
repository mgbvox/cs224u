import pytest
import torch.utils.data
from huggingface_hub import HfApi, ModelFilter
from torch import nn
from datasets import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

from custom_models.sentiment.mixture_of_experts import Expert, ExpertMixture


@pytest.fixture(scope="session")
def global_model() -> ExpertMixture:
    em = ExpertMixture()
    return em


@pytest.fixture(scope="session")
def global_dataset() -> Dataset:
    dynasent_r1 = load_dataset("dynabench/dynasent", "dynabench.dynasent.r1.all")
    return dynasent_r1


def get_model_names(n: int) -> list[str]:
    hf = HfApi()
    models = hf.list_models(
        filter=ModelFilter(
            task="text-classification",
            library="pytorch",
        ),
        limit=100,
    )
    return [m.id for m in models][:n]


def test_lazy_encode(global_dataset, global_model):
    graph = global_model.build_graph()
    data = global_model.build_dataset(global_dataset)
