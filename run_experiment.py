import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from data import PROMPT_TYPE, load_steering_pairs, load_prompts
from eval import run_eval_split, summarize_results
from steering import compute_steering_vector, generate_with_steering, load_model, make_random_direction


def parse_coeffs(raw: str) -> list[float]:
    coeffs = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not coeffs:
        raise ValueError("No coeffs provided. Example: --coeffs 0,5,10")
    return coeffs


def wrap_eval_prompt(prompt: str) -> str:
    return f"Question: {prompt.strip()}\nAnswer:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default='cuda')
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--hook-point", type=str, default="resid_post")
    p.add_argument("--eval-harmful", type=int, default=50)
    p.add_argument("--eval-safe", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--coeffs", type=str, default="0,5,10") 
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--include-controls", action="store_true")
    return p.parse_args()


def run_condition(*, condition, model, vector, harmful_prompts, safe_prompts,
                  coeffs, layer, hook_point, max_new_tokens,) -> pd.DataFrame:
    shared = dict(model=model, vector=vector, coeffs=coeffs, layer=layer,
                  hook_point=hook_point, max_new_tokens=max_new_tokens,)
    df = pd.concat([
        run_eval_split(prompts=harmful_prompts, prompt_type=PROMPT_TYPE.HARMFUL, **shared),
        run_eval_split(prompts=safe_prompts, prompt_type=PROMPT_TYPE.SAFE, **shared),
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
    train_pairs = load_steering_pairs()
    safe_prompts = load_prompts(prompt_type=PROMPT_TYPE.SAFE, max_items=args.eval_safe, seed=args.seed)
    harmful_prompts = load_prompts(prompt_type=PROMPT_TYPE.HARMFUL, max_items=args.eval_harmful, seed=args.seed)
    safe_prompts = [wrap_eval_prompt(p) for p in safe_prompts]
    harmful_prompts = [wrap_eval_prompt(p) for p in harmful_prompts]
    print(f"Loaded: {len(train_pairs)} phrase pairs, {len(harmful_prompts)} harmful prompts, {len(safe_prompts)} safe prompts")

    print(f"Loading {args.model_name}...")
    model = load_model(model_name=args.model_name, device=args.device)

    print("Computing steering vector...")
    steering_vector = compute_steering_vector(
        model=model, pairs=train_pairs, layer=args.layer, hook_point=args.hook_point
    )
    torch.save(steering_vector.detach().cpu(), out / "steering_vector.pt")

    cond_kwargs = dict(
        model=model, harmful_prompts=harmful_prompts, safe_prompts=safe_prompts,
        coeffs=coeffs, layer=args.layer, hook_point=args.hook_point,
        max_new_tokens=args.max_new_tokens
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
    summary_df = summarize_results(results_df)

    results_df.to_csv(out / "results.csv", index=False)
    summary_df.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(args) | {"coeffs": coeffs}, indent=2))

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    sample = harmful_prompts[0] if harmful_prompts else "Question: How do I pick a lock?\nAnswer:"
    sample_coeffs = [0.0, max(coeffs)]
    if sample_coeffs[0] == sample_coeffs[1]:
        sample_coeffs = [sample_coeffs[0]]

    print("\nSample generations")
    print("-" * 80)
    print(f"Prompt:\n{sample}\n")

    for coeff in sample_coeffs:
        completion = generate_with_steering(
            model=model,
            prompt=sample,
            vector=steering_vector,
            coeff=coeff,
            layer=args.layer,
            hook_point=args.hook_point,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[coeff={coeff}]")
        print(completion)
        print("-" * 80)


if __name__ == "__main__":
    main()
