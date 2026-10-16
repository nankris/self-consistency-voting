"""Qwen3-4B non-thinking self-consistency on Colab. Resumable per 10-question batch.

This process does not score accuracy. It only writes generations.
"""
from __future__ import annotations

import json
import os
import time
import traceback

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ROOT = "/content/035r3"
PROTOCOL_PATH = ROOT + "/protocol.json"


def write_progress(**fields) -> None:
    path = ROOT + "/progress.json"
    current = {}
    if os.path.exists(path):
        try:
            current = json.loads(open(path).read())
        except json.JSONDecodeError:
            current = {}
    current.update(fields)
    current["updated"] = time.time()
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(current, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_done() -> set[int]:
    done = set()
    folder = ROOT + "/batches"
    if not os.path.isdir(folder):
        return done
    for name in os.listdir(folder):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(folder, name)) as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["pos"])
    return done


def prepare_questions(protocol: dict) -> list[dict]:
    path = ROOT + "/questions.json"
    if os.path.exists(path):
        return json.loads(open(path).read())
    import random

    from datasets import load_dataset

    data = load_dataset(protocol["dataset"], protocol["dataset_config"], split=protocol["dataset_split"])
    ids = list(range(len(data)))
    random.Random(int(protocol["shuffle_seed"])).shuffle(ids)
    n = int(protocol["n_questions"])
    questions = []
    for pos, index in enumerate(ids[:n]):
        row = data[index]
        gold = row["answer"].split("####")[-1].strip()
        questions.append(
            {
                "pos": pos,
                "index": index,
                "question": row["question"],
                "answer": row["answer"],
                "gold": gold,
            }
        )
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(questions, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return questions


def load_llm(dtype: str):
    import torch
    from vllm import LLM

    attempts = [
        dict(max_model_len=4096, max_num_seqs=64, gpu_memory_utilization=0.90),
        dict(max_model_len=3072, max_num_seqs=32, gpu_memory_utilization=0.92),
        dict(max_model_len=2560, max_num_seqs=16, gpu_memory_utilization=0.95),
    ]
    last = None
    for config in attempts:
        try:
            llm = LLM(
                model="Qwen/Qwen3-4B",
                dtype=dtype,
                seed=35035,
                trust_remote_code=False,
                disable_log_stats=True,
                **config,
            )
            return llm, config, torch.cuda.get_device_name()
        except Exception as exc:  # retry only apparent OOM during engine init
            last = exc
            message = str(exc).lower()
            if "memory" not in message and "oom" not in message and "cuda" not in message:
                raise
            print("LLM init failed, trying a smaller cache:", exc, flush=True)
    raise last


def main() -> None:
    protocol = json.loads(open(PROTOCOL_PATH).read())
    config = json.loads(open(ROOT + "/config.json").read()) if os.path.exists(ROOT + "/config.json") else {}
    dtype = config.get("dtype", "bfloat16")
    batch_size = int(config.get("batch_size", 10))
    deadline = float(config.get("deadline_epoch", time.time() + 3 * 3600))
    os.makedirs(ROOT + "/batches", exist_ok=True)
    questions = prepare_questions(protocol)
    done = load_done()
    write_progress(status="prepared", completed=len(done), n=len(questions), files=batch_names())

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    import importlib.metadata

    tokenizer = AutoTokenizer.from_pretrained(protocol["model"])
    example = tokenizer.apply_chat_template(
        [{"role": "user", "content": questions[0]["question"] + protocol["prompt_suffix"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    meta = {
        "model": protocol["model"],
        "dtype": dtype,
        "thinking": False,
        "n_questions": len(questions),
        "prompt_suffix": protocol["prompt_suffix"],
        "prompt_tail": example[-240:],
        "protocol": protocol,
        "start": time.time(),
        "versions": {},
    }
    for package in ("vllm", "transformers", "torch", "datasets"):
        try:
            meta["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta["versions"][package] = None
    json.dump(meta, open(ROOT + "/metadata.json", "w"))
    write_progress(status="loading", completed=len(done), n=len(questions), prompt_tail=example[-240:])

    llm, engine, gpu_name = load_llm(dtype)
    meta["gpu"] = gpu_name
    meta["engine"] = engine
    json.dump(meta, open(ROOT + "/metadata.json", "w"))
    write_progress(status="running", completed=len(done), n=len(questions), gpu=gpu_name)

    suffix = protocol["prompt_suffix"]
    n_samples = int(protocol["n_samples"])
    temperature = float(protocol["temperature"])
    top_p = float(protocol["top_p"])
    top_k = int(protocol["top_k"])
    max_tokens = int(protocol["max_tokens"])
    seed_base = int(protocol["sample_seed_base"])

    for start in range(0, len(questions), batch_size):
        if time.time() > deadline:
            write_progress(status="budget_stop", completed=len(load_done()), n=len(questions), files=batch_names())
            meta["end"] = time.time()
            meta["stop_reason"] = "deadline"
            json.dump(meta, open(ROOT + "/metadata.json", "w"))
            print("BUDGET_STOP", flush=True)
            return
        batch = [q for q in questions[start : start + batch_size] if q["pos"] not in done]
        if not batch:
            continue
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q["question"] + suffix}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for q in batch
        ]
        params = [
            SamplingParams(
                n=n_samples,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                seed=seed_base + q["pos"],
                skip_special_tokens=True,
            )
            for q in batch
        ]
        t0 = time.time()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        seconds = time.time() - t0
        records = []
        for question, output in zip(batch, outputs):
            records.append(
                {
                    "pos": question["pos"],
                    "index": question["index"],
                    "gold": question["gold"],
                    "thinking": False,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_tokens": max_tokens,
                    "seed": seed_base + question["pos"],
                    "batch_seconds": seconds,
                    "samples": [
                        {
                            "text": sample.text,
                            "tokens": len(sample.token_ids),
                            "finish_reason": sample.finish_reason,
                        }
                        for sample in output.outputs
                    ],
                }
            )
        name = f"b{start:04d}.jsonl"
        dest = os.path.join(ROOT, "batches", name)
        tmp = dest + ".tmp"
        with open(tmp, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dest)
        done.update(record["pos"] for record in records)
        lengths = [s["finish_reason"] == "length" for record in records for s in record["samples"]]
        write_progress(
            status="running",
            completed=len(done),
            n=len(questions),
            files=batch_names(),
            gpu=gpu_name,
            last_batch=name,
            last_batch_seconds=seconds,
            last_batch_truncation_rate=sum(lengths) / len(lengths),
        )
        print(
            f"BATCH {len(done)}/{len(questions)} {name} {seconds:.1f}s trunc={sum(lengths)}/{len(lengths)}",
            flush=True,
        )

    meta["end"] = time.time()
    meta["stop_reason"] = "complete"
    json.dump(meta, open(ROOT + "/metadata.json", "w"))
    write_progress(status="complete", completed=len(done), n=len(questions), files=batch_names(), gpu=gpu_name)
    print("COMPLETE", len(done), flush=True)


def batch_names() -> list[str]:
    folder = ROOT + "/batches"
    if not os.path.isdir(folder):
        return []
    return sorted(name for name in os.listdir(folder) if name.endswith(".jsonl"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        os.makedirs(ROOT, exist_ok=True)
        open(ROOT + "/error.txt", "w").write(err)
        write_progress(status="error", error_tail=err[-2000:])
        print(err, flush=True)
        raise
