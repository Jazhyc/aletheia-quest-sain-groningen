#!/usr/bin/env python3
"""Compare neutral fact audits from Qwen organisms and their base model."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.blackbox.audit_gpt_oss_shared_errors import (
    RESULTS,
    audit_prompt,
    load_examples,
    shared_false_negatives,
)


MODEL = "Qwen/Qwen3.5-27B"
METHOD = "qwen27b_organism_self_query_shared_errors_v1"
VERDICT_RE = re.compile(
    r"(?i)VERDICT\s*:\s*(CONTAINS_FALSE_DETAIL|NO_FALSE_DETAIL|UNCERTAIN)"
)


def render_prompt(tokenizer: Any, example: dict[str, Any]) -> str:
    prompt = audit_prompt(example, privileged=False)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def verdict(text: str) -> str | None:
    matches = VERDICT_RE.findall(text)
    return matches[-1].upper() if matches else None


def main() -> None:
    examples = load_examples(shared_false_negatives())
    if any(example.get("model") != MODEL for example in examples):
        unexpected = sorted({example.get("model") for example in examples})
        raise RuntimeError(f"expected only {MODEL}, got {unexpected}")
    if any(not example.get("lora") for example in examples):
        raise RuntimeError("the selected shared false negatives must all have organism LoRAs")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = [render_prompt(tokenizer, example) for example in examples]
    lora_repos = sorted({str(example["lora"]) for example in examples})
    lora_paths = {
        repo: Path(snapshot_download(repo_id=repo))
        for repo in lora_repos
    }

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        max_num_seqs=64,
        enable_lora=True,
        max_lora_rank=128,
        max_loras=1,
        max_cpu_loras=max(1, len(lora_repos)),
    )
    sampling = SamplingParams(max_tokens=1024, temperature=0.0)
    started = time.time()

    base_outputs = llm.generate(prompts, sampling)
    base_text = [output.outputs[0].text if output.outputs else "" for output in base_outputs]

    organism_text: list[str | None] = [None] * len(examples)
    for lora_id, repo in enumerate(lora_repos, 1):
        offsets = [i for i, example in enumerate(examples) if example["lora"] == repo]
        request = LoRARequest(repo, lora_id, lora_paths[repo].as_posix())
        outputs = llm.generate([prompts[i] for i in offsets], sampling, lora_request=request)
        for offset, output in zip(offsets, outputs, strict=True):
            organism_text[offset] = output.outputs[0].text if output.outputs else ""

    elapsed = time.time() - started
    output_dir = RESULTS / METHOD
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for example, prompt, base, organism in zip(
        examples, prompts, base_text, organism_text, strict=True
    ):
        if organism is None:
            raise RuntimeError(f"missing organism output for {example['dataset']} {example['index']}")
        records.append({
            **example,
            "prompt": prompt,
            "base_completion": base,
            "base_verdict": verdict(base),
            "organism_completion": organism,
            "organism_verdict": verdict(organism),
        })
    (output_dir / "generations.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )
    result = {
        "method": METHOD,
        "model": MODEL,
        "rows": len(records),
        "lora_repos": lora_repos,
        "prompt_evaluations": 2 * len(records),
        "score_seconds": elapsed,
        "rows_per_second": 2 * len(records) / elapsed,
        "base_verdicts": dict(_counts(record["base_verdict"] for record in records)),
        "organism_verdicts": dict(_counts(record["organism_verdict"] for record in records)),
        "verdict_disagreements": sum(
            record["base_verdict"] != record["organism_verdict"] for record in records
        ),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def _counts(values: Any) -> list[tuple[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value or "PARSE_ERROR")] += 1
    return sorted(counts.items())


if __name__ == "__main__":
    main()
