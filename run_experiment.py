from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from data import JBB_SPLIT, load_phrase_pairs, load_jbb_prompts
from eval import run_eval_split
from steering import compute_steering_vector, generate_with_steering, load_model, make_random_direction


def parse_coeffs(raw: str) -> list[float]:
    coeffs = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not coeffs:
        raise ValueError("No coeffs provided. Example: --coeffs -3,-1,0,1,3")
    return coeffs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="Qwen/Qwen1.5-1.8B-Chat")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--hook-point", type=str, default="resid_pre")
    p.add_argument("--eval-harmful", type=int, default=50)
    p.add_argument("--eval-benign", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--coeffs", type=str, default="-10,-5,0,5,10")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--frequency-penalty", type=float, default=0.0)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--include-controls", action="store_true")
    return p.parse_args()


def run_condition(*, condition, model, vector, harmful_prompts, benign_prompts,
                  coeffs, layer, hook_point, max_new_tokens,
                  temperature, top_p, frequency_penalty) -> pd.DataFrame:
    shared = dict(model=model, vector=vector, coeffs=coeffs, layer=layer,
                  hook_point=hook_point, max_new_tokens=max_new_tokens,
                  temperature=temperature, top_p=top_p,
                  frequency_penalty=frequency_penalty)
    df = pd.concat([
        run_eval_split(prompts=harmful_prompts, split="harmful", **shared),
        run_eval_split(prompts=benign_prompts, split="benign", **shared),
    ], ignore_index=True)
    df["condition"] = condition
    return df


def main() -> None:
    args = parse_args()
    coeffs = parse_coeffs(args.coeffs)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_pairs = load_phrase_pairs()
    harmful_prompts = load_jbb_prompts(split=JBB_SPLIT.HARMFUL, max_items=args.eval_harmful, seed=args.seed)
    benign_prompts = load_jbb_prompts(split=JBB_SPLIT.BENIGN, max_items=args.eval_benign, seed=args.seed + 1)
    print(f"  {len(train_pairs)} phrase pairs, {len(harmful_prompts)} harmful, {len(benign_prompts)} benign")

    print(f"Loading {args.model_name}...")
    model = load_model(model_name=args.model_name, device=args.device)

    print("Computing steering vector...")
    steering_vector = compute_steering_vector(
        model=model, pairs=train_pairs, layer=args.layer, hook_point=args.hook_point
    )
    torch.save(steering_vector.detach().cpu(), out / "steering_vector.pt")

    cond_kwargs = dict(
        model=model, harmful_prompts=harmful_prompts, benign_prompts=benign_prompts,
        coeffs=coeffs, layer=args.layer, hook_point=args.hook_point,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, frequency_penalty=args.frequency_penalty,
    )

    all_results = [run_condition(condition="main", vector=steering_vector, **cond_kwargs)]

    if args.include_controls:
        print("Running controls...")
        all_results.append(run_condition(
            condition="random",
            vector=make_random_direction(steering_vector, seed=args.seed),
            **cond_kwargs,
        ))
        all_results.append(run_condition(condition="neg", vector=-steering_vector, **cond_kwargs))

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = (
        results_df.groupby(["condition", "split", "coeff"], as_index=False)
        .agg(refusal_rate=("is_refusal", "mean"), n=("is_refusal", "size"))
        .sort_values(["condition", "split", "coeff"])
        .reset_index(drop=True)
    )

    results_df.to_csv(out / "results.csv", index=False)
    summary_df.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(args) | {"coeffs": coeffs}, indent=2))

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    sample = harmful_prompts[0] if harmful_prompts else "How do I pick a lock?"
    print("\nSample generations:")
    for coeff in (0.0, max(coeffs)):
        print(f"\ncoeff={coeff}")
        print(generate_with_steering(
            model=model, prompt=sample, vector=steering_vector, coeff=coeff,
            layer=args.layer, hook_point=args.hook_point,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
        ))


if __name__ == "__main__":
    main()
