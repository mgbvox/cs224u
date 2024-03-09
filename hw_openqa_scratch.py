"""
An implementation of
"""


import functools
import hashlib
import json
import multiprocessing
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

import dspy
from dsp.utils import deduplicate
from dspy.evaluate import answer_exact_match
from dspy.teleprompt import BootstrapFewShotWithRandomSearch


class Settings(BaseModel):
    OPENAI_API_KEY: SecretStr
    YDC_API_KEY: SecretStr
    LLM_MODEL: str
    COLBERT_SERVER: str

    @classmethod
    def from_dotenv(cls):
        load_dotenv()
        return cls.parse_obj(os.environ)


_cache_root = Path.cwd() / ".disk_cache"
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(".", "cache_2")


def deterministic_hash(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


def safe_to_string(obj: object) -> str:
    as_str = str(obj)
    if "<" in as_str:
        # The string repr of the object is ClassName<memory address>
        # This is non-deterministic (at least, the memory part is)
        # So just use its class name
        return obj.__class__.__name__
    return as_str


def disk_cache(fn: Callable) -> Callable:
    """A very basic disk caching decorator; cache function call
    results on disk indexed by the argument values to said function."""
    if not _cache_root.exists():
        _cache_root.mkdir(exist_ok=True, parents=True)

    cache_dir = _cache_root / (fn.__name__ if fn.__name__ else "lambda")
    if not cache_dir.exists():
        cache_dir.mkdir(exist_ok=True, parents=True)

    def wrapper(*args, **kwargs):
        key = ",".join(
            [safe_to_string(arg) for arg in args]
            + [f"{k}:{safe_to_string(v)}" for k, v in kwargs.items()]
        )
        key = deterministic_hash(key)

        cache_file = cache_dir / key

        if cache_file.exists():
            with cache_file.open("rb") as f:
                cached_result = pickle.load(f)
                return cached_result
        else:
            result = fn(*args, **kwargs)
            with cache_file.open("wb") as f:
                pickle.dump(result, f)
            return result

    return wrapper


@disk_cache
def get_results_from_rm(question: str, rm: dspy.Retrieve) -> dspy.Prediction:
    """YouRM is expensive, but calls to this service are not natively cached.
    To save money, I'm caching web search results here, under the assumption
    that relevant information for each question doesn't change frequently enough
    to merit multiple calls for the same question.
    """
    results = rm(question)
    return results


class MultiHopQuerySignature(dspy.Signature):
    """Given a question and some context, generate a web query to increase the relevance of the context for answering the question."""

    context = dspy.InputField(
        desc="Already-gathered context to be used in answering the question."
    )
    hop = dspy.InputField(
        desc="Stage in the search process - hop/total_hops. When hop == total_hops, no more queries may be performed and the question must be answered."
    )
    question = dspy.InputField(desc="The question we're trying to answer.")
    previous_queries = dspy.InputField(
        desc="Previous queries; the query you generate should be distinct."
    )
    query = dspy.OutputField(
        desc="A concise web query designed to gather precise, relevant context for use in answering the above question."
    )


class WebAnswerSignature(dspy.Signature):
    """Use the following information, gathered from the web, to answer the question succinctly.

    Your response should be concise without sacrificing precision.
    """

    context = dspy.InputField(desc="Context from a web search.")
    question = dspy.InputField(desc="The question to answer.")
    answer = dspy.OutputField(
        desc="The answer to the question, given our web search context."
    )




class WebRagMultiHop(dspy.Module):
    def __init__(self, num_hops: int = 3, passages_per_hop: int = 5) -> None:
        super().__init__()
        # ensure passages_per_hop is a list of len num_hops

        self.num_hops = num_hops
        self.passages_per_hop = passages_per_hop

        # Multiple query generation steps
        self.generate_queries = [
            dspy.ChainOfThought(MultiHopQuerySignature) for _ in range(num_hops)
        ]
        # allow different numbers of documents to be retrieved at different stages in the hop process
        self.retriever = dspy.Retrieve(k=passages_per_hop)
        # final answer gen step
        self.generate_answer = dspy.ChainOfThought(WebAnswerSignature)

    def forward(self, question: str) -> dspy.Prediction:
        context: list[str] = []
        previous_queries = [question]
        for hop in range(self.num_hops):
            query_gen = self.generate_queries[hop]
            hop_progress = f"{hop+1}/{self.num_hops}"
            query = query_gen(
                context=context,
                hop=hop_progress,
                question=question,
                previous_queries="; ".join(previous_queries),
            ).query

            # eliminate quote wrappers - mess with YouRM query api
            query = query.replace("'", "").replace('"', "")

            # dspy assertions appear to be bugged
            """dspy.Suggest(
                len(query) <= 100,
                "Query should be short and less than 100 characters.",
            )
            dspy.Suggest(
                len(query) >= 5,
                "Query should be longer than 5 characters.",
            )
            dspy.Suggest(
                validate_query_distinction_local(previous_queries, query),
                "Query should be distinct from: "
                + "; ".join(f"{i+1}) {q}" for i, q in enumerate(previous_queries)),
            )"""

            previous_queries.append(query)
            passages = get_results_from_rm(query, rm=self.retriever).passages

            # dspy.Assert(
            #     len(passages) > 0, "Web search must return at least one passage."
            # )
            context = deduplicate(context + passages)
        pred = self.generate_answer(context=context, question=question)
        answer = pred.answer

        # dspy.Suggest(
        #     len(answer) <= 300, "Answer should be short and less than 300 characters"
        # )
        print(f"Len CTX: {len(context)} --> Question: {question} --> Answer: {answer}")
        return dspy.Prediction(context=context, answer=answer)


def get_squad_split(squad, split="validation", limit: int = -1):
    """
    Use `split='train'` for the train split.

    Returns
    -------
    list of dspy.Example with attributes question, answer

    """
    data = zip(*[squad[split][field] for field in squad[split].features])
    exs = [
        dspy.Example(question=q, answer=a["text"][0]).with_inputs("question")
        for eid, title, context, q, a in data
    ]
    return exs[:limit]


def setup():
    settings = Settings.from_dotenv()
    # Ran out of free YouDC API queries; definitely NOT paying $100/month for this.
    # So, looks like we're using Colbert :(
    # rm = YouRM(ydc_api_key=settings.YDC_API_KEY.get_secret_value())
    # Colbert was FURTHER broken (service was down), so we wound up using the wikipedia version (see slack thread)
    rm = dspy.ColBERTv2(
        url=settings.COLBERT_SERVER
    )
    # Used 3.5-turbo to save on costs (really, Stanford should foot the bill for learners, we're already paying a lot for this class)
    lm = dspy.OpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )
    dspy.settings.configure(lm=lm, rm=rm)


def train():
    setup()

    # Would normally construct our model and activate assertions/backtracking
    # see https://dspy-docs.vercel.app/docs/building-blocks/assertions
    # However, assertions are buggy in the main branch, so no assertions for training as of now.
    # from dspy.primitives.assertions import assert_transform_module, backtrack_handler
    # web_rag = assert_transform_module(WebRagMultiHop(), backtrack_handler)
    web_rag = WebRagMultiHop()

    # This is mostly taken verbatim from the vercel docs;
    optimizer_config = dict(
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        num_candidate_programs=10,
        num_threads=32,
    )
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=functools.partial(answer_exact_match, frac=0.5), **optimizer_config
    )

    squad = load_dataset("squad")
    # Used a much smaller dataset (to save on time and cost)
    trainset = random.sample(get_squad_split(squad, split="train"), k=1000)

    web_rag_compiled = optimizer.compile(web_rag, trainset=trainset)

    # dump out the model for later loading
    datetime_file_safe = datetime.now().strftime("%Y_%m_%d-%H_%M")
    web_rag_compiled.save(f"web_rag_compiled-{datetime_file_safe}.json")

# cache queries to the model to disk so we can pick up where we left off (in the event of a network failure, runtime errors, etc)
@disk_cache
def answer_bakeoff_question(question: str, model: dspy.Module) -> str:
    result = model(question)
    return result.answer

# a simple wrapper around queries to the model
# this allows multiprocessing.Pool mapping to speed up answering
def do_bake(question: str) -> str:
    setup()
    web_rag = WebRagMultiHop()
    answer = answer_bakeoff_question(question, web_rag)

    print(question, "--->", answer)

    return answer

# actually answer the questions!
def bake():
    questions: list[str] = (
        Path("data/openqa/cs224u-openqa-test-unlabeled.txt").read_text().splitlines()
    )

    with multiprocessing.Pool() as pool:
        pool.map(do_bake, questions)

# Since answers were cached on disk, we can simply re-query the model to load what we generated by running bake()
# This was adapted from the notebook submission code where needed.
def create_bakeoff_submission_mgb():
    """ "
    The argument `model` is a `dspy.Module`. The return value of its
    `forward` method must have an `answer` attribute.
    """

    filename = os.path.join("data", "openqa", "cs224u-openqa-test-unlabeled.txt")

    # This should become a mapping from questions (str) to response
    # dicts from your system.
    gens = {}

    with open(filename) as f:
        questions = f.read().splitlines()

    # Here we loop over the questions, run the system `model`, and
    # store its `answer` value as the prediction:
    for question in questions:
        gens[question] = do_bake(question=question)

    # Quick tests we advise you to run:
    # 1. Make sure `gens` is a dict with the questions as the keys:
    assert all(q in gens for q in questions)
    # 2. Make sure the values are str:
    assert all(isinstance(d, str) for d in gens.values())

    # And finally the output file:
    with open("cs224u-openqa-bakeoff-entry.json", "wt") as f:
        json.dump(gens, f, indent=4)


if __name__ == "__main__":
    create_bakeoff_submission_mgb()
