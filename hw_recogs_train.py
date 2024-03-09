"""
Preamble:
As was the case for last PSet, this project resulted in several PRs
to the dspy repo:
https://github.com/stanfordnlp/dspy/pull/467 (Updated)

HF Models + Tokens:
https://github.com/stanfordnlp/dspy/pull/611
I initially intended to use an ensemble of Alpaca-like
models for this project, but they proved too large to run
on the computational resources I had. Still, I added
token auth to dspy.HFModels.

Also, added some dspy issues:
https://github.com/stanfordnlp/dspy/issues/587
https://github.com/stanfordnlp/dspy/issues/589

Finally, began compiling and publishing my utils in a standalone
package:
https://github.com/mgbvox/cs224u-utils

This will be updated with other useful tools as I build them.

Writeup:
BOOTSTRAP FINETUNED DSPY MODEL FOR RECOGS

This is the final iteration of my attempts at writing a passable DSPY module
for the Recogs task. In short, it's a chain-of-thought module that
reasons about the procedure for creating recogs formalisms
before translating an input sentence into an output sentence.

Originally, I had a secondary step in the training process whereby a sample
of the train dataset was given to a chain-of-thought module with the signature:

class FormInferrenceSignature(dspy.Signature):
    '''Given sentences and their logical forms, produce a ruleset
    describing the procedure by which each logical form is produced.'''

    # a list of dspy.Examples transformed into ${sentence} -> ${logicalForm}
    context = dspy.InputField()
    # derived logical transformation rules
    rules = dspy.OutputField()

The rationale here was to force the LLM to try and deduce the rules for translating
ahead of time, which would hopefully improve the subsequent COT inferrence.

This secondary module underwent its own bootstrapping process and the derived roles were passed
into our main model as an adjuvant to more chain-of-thought reasoning:

class RecogsSignature(dspy.Signature):
    '''Translate english sentences into their logical form.'''

    sentence = dspy.InputField()
    # derived logical transformation rules passed in here
    rules = dspy.InputField(
        desc="These rules were deduced ahead of time, and may or may not be correct. Think critically about them."
    )
    logical_form = dspy.OutputField()

THIS WAS ULTIMATELY ABANDONED as it wasn't increasing performance and more than
doubled my compute spend.

The final system, below, is essentially the above minus the rules-extraction
step.

Inference proved slow (and expensive) - after 900ish evals on the gen set, I'd
spent >$50 on OpenAI's API, so killed it early. I'll submit to the bakeoff for completeness'
sake, but don't expect to win. Alas.
"""

import multiprocessing
import os
from typing import Literal, Optional

import numpy as np
import pandas as pd

# install from: https://github.com/mgbvox/cs224u-utils
from cs224u_utils.cache import disk_cache
from dotenv import load_dotenv
from tqdm import tqdm

import dspy
from compgen import recogs_exact_match
from dspy.teleprompt import LabeledFewShot

SRC_DIRNAME = os.path.join("data", "recogs")


def load_split(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, delimiter="\t", names=["input", "output", "category"])


def get_dataset(
    split_name: Literal["train", "dev", "gen"],
    k: Optional[int] = None,
) -> list[dspy.Example]:
    dataset = load_split(f"{SRC_DIRNAME}/{split_name}.tsv")
    if k:
        dataset = dataset.sample(n=k)

    dspy_recogs_train = [
        dspy.Example(sentence=row["input"], logical_form=row["output"]).with_inputs(
            "sentence"
        )
        for _, row in tqdm(dataset.iterrows())
    ]

    return dspy_recogs_train


def setup():
    load_dotenv()
    lm = dspy.OpenAI(
        model="gpt-4-turbo-preview",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=2048,
    )

    dspy.settings.configure(lm=lm)


class RecogsSignature(dspy.Signature):
    """Translate english sentences into their logical form.

    Note that definite articles are translated into asterisks, e.g:

    The dog -> * dog (some number)
    """

    sentence = dspy.InputField(desc="The sentence to translate.")
    logical_form = dspy.OutputField(desc="The logical form of the sentence.")


class RecogsDspy(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(RecogsSignature)

    def forward(self, sentence: str):
        pred = self.predict(sentence=sentence)
        return pred


def recogs_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None):
    # dspy.settings.lm.inspect_history(1)
    print()
    print(gold.sentence)
    print(gold.logical_form)
    print(pred.logical_form)
    print("------------------")
    out = recogs_exact_match(gold.logical_form, pred.logical_form)
    print("Equivalent:", out)
    return int(out)


def train():
    ds = get_dataset("train", k=100)
    val = get_dataset("dev", k=30)

    model = RecogsDspy()
    bfs = dspy.teleprompt.BootstrapFewShotWithRandomSearch(
        metric=recogs_metric,
        max_bootstrapped_demos=8,
        max_labeled_demos=8,
        num_threads=16,
    )
    model = bfs.compile(model, trainset=ds, valset=val)

    model.save("recogs_model_bootstrap_random.json")


@disk_cache
def do_eval(input: dspy.Example, model: RecogsDspy, eta: str) -> str:
    _ = eta
    out = model(sentence=input.sentence)
    print()
    print(out)
    print("--------------")
    return out.logical_form


def eval_worker(item: dspy.Example):
    setup()
    model = RecogsDspy()
    model.load("recogs_model_bootstrap_random.json")
    return do_eval(item, model=model, eta="asdf")


def run_eval():
    bakeoff_df = pd.read_csv(
        os.path.join(SRC_DIRNAME, "cs224u-recogs-test-unlabeled.tsv"),
        sep="\t",
        index_col=0,
    )
    to_solve = bakeoff_df.input.apply(
        lambda x: dspy.Example(sentence=x).with_inputs("sentence")
    ).values
    with multiprocessing.Pool() as pool:
        out = list(tqdm(pool.imap(eval_worker, to_solve), total=len(to_solve)))
    print(out)


def fill_submission():
    setup()
    # ds = get_dataset("gen")
    df = pd.read_csv(
        os.path.join(SRC_DIRNAME, "cs224u-recogs-test-unlabeled.tsv"),
        sep="\t",
        index_col=0,
    )

    to_solve = df.input.apply(
        lambda x: dspy.Example(sentence=x).with_inputs("sentence")
    ).values

    out = [eval_worker(item) for item in tqdm(to_solve)]
    df["prediction"] = out
    df.to_csv("cs224u-recogs-bakeoff-entry.tsv", sep="\t")


if __name__ == "__main__":
    fill_submission()
    # run_eval()
