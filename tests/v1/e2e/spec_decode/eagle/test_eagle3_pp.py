# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EAGLE3 speculative decoding under pipeline parallelism."""

import pytest
import torch

from tests.utils import multi_gpu_test
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.metrics.reader import Metric

PROMPTS = [
    "The capital of France is",
    "2 + 2 equals",
    "In one word, the color of the sky is",
    "Q: If a train travels 60 miles in 1.5 hours, what is its average speed?\nA:",
]

# The taps arrive over the pipeline handoff, which is deterministic, so PP=2
# acceptance tracked PP=1 exactly in development. The margin covers scheduling
# jitter while staying well inside the ~11% drop a dropped tap produced.
ACCEPTANCE_TOLERANCE = 0.95


def _acceptance_length(metrics: list[Metric]) -> float:
    """1 + accepted/drafts, the mean tokens emitted per target forward."""
    by_name = {m.name: m for m in metrics}
    drafts = by_name.get("vllm:spec_decode_num_drafts")
    accepted = by_name.get("vllm:spec_decode_num_accepted_tokens")
    assert drafts is not None and accepted is not None, (
        "spec_decode metrics missing; check disable_log_stats=False"
    )
    assert int(drafts.value) > 0, "drafter never proposed anything"
    return 1.0 + int(accepted.value) / int(drafts.value)


def _run(pp_size: int, model: str, draft: str, cudagraph_mode: str | None) -> float:
    kwargs = dict(
        model=model,
        tensor_parallel_size=1,
        pipeline_parallel_size=pp_size,
        max_model_len=512,
        gpu_memory_utilization=0.45,
        disable_log_stats=False,
        speculative_config={
            "method": "eagle3",
            "model": draft,
            "num_speculative_tokens": 3,
        },
    )
    if cudagraph_mode is not None:
        kwargs["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    llm = LLM(**kwargs)
    try:
        llm.generate(
            PROMPTS,
            SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True),
        )
        return _acceptance_length(llm.get_metrics())
    finally:
        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model,draft",
    [
        (
            "meta-llama/Llama-3.2-1B-Instruct",
            "nm-testing/Llama3_2_1B_speculator.eagle3",
        ),
    ],
)
@pytest.mark.parametrize("cudagraph_mode", [None, "FULL_AND_PIECEWISE"])
def test_eagle3_pipeline_parallel_acceptance(
    model: str,
    draft: str,
    cudagraph_mode: str | None,
):
    """Aux hidden states must survive the pipeline handoff.

    Compares acceptance length at PP=2 against PP=1 on the same model. This
    feature fails quietly: a stale, mis-ordered or dropped tap still yields
    well-formed proposals, so the engine boots and answers -- the proposals just
    get rejected more often. Acceptance is what detects that.

    Greedy text parity deliberately is not asserted. bf16 argmax ties break
    differently once the batch shape changes, so even two runs with spec decode
    off diverge from each other, which would make such an assertion flaky for
    reasons unrelated to this feature.
    """
    baseline = _run(1, model, draft, cudagraph_mode)
    parallel = _run(2, model, draft, cudagraph_mode)

    assert parallel >= baseline * ACCEPTANCE_TOLERANCE, (
        f"acceptance length regressed under PP=2 "
        f"(cudagraph_mode={cudagraph_mode}): "
        f"{parallel:.3f} < {baseline:.3f} * {ACCEPTANCE_TOLERANCE}"
    )
