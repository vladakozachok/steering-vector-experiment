# Steering Vector Experiment

Activation steering experiment with TransformerLens for refusal behavior on Qwen/Qwen1.5-1.8B-Chat model.

## Implementation

### Steering data (contrastive pairs)

`data.py` defines `STEERING_PROMPTS` with 14 safety categories spanning animal abuse, child abuse, hate speech, self-harm, terrorism, and more (as defined in the PKU-Alignment/BeaverTails-Evaluation) dataset.

Each steering pair is built as:

```text
Question: <question>
Choices:
(A) <choice text>
(B) <choice text>
Answer: <letter>
```

The positive/negative pair differs only at the final answer letter token.
To avoid learning only an `A vs B` token-identity direction, `load_steering_pairs()` counterbalances labels:
half the prompts use `A=refusal, B=harmful`, and half use `A=harmful, B=refusal`.

### Steering vector

In `compute_steering_vector` (`steering.py`):

1. Run both texts in each pair through the model.
2. Read the residual stream activation at `blocks.{layer}.hook_{hook_point}` at the **final token position** which shows the choice between the harmful and refusal responses.
3. Compute `diff = pos_activation - neg_activation` per pair.
4. Average diffs and L2-normalize.

Formula:

```
v = normalize(mean_i(a_pos_i - a_neg_i))
```

### Injection

In `generate_with_steering` (`steering.py`):

- Generation is greedy (`do_sample=False`).
- The steering vector is added to **all token positions** at every forward pass (prefill and decode):
  `resid += coeff * v`

### Evaluation data

I use two datasets for evaluating the steering:
- `PKU-Alignment/BeaverTails-Evaluation`: used to generate responses to harmful prompts and evaluate refusal rates.
- `databricks/databricks-dolly-15k`: used to explore responses to regular safe prompts and evaluate whether the steering affects only harmful refusal or the overall refusal. 

Eval prompts are wrapped before generation:

```text
Question: <prompt>
Answer:
```

### Refusal metric

`eval.py` marks a completion as refusal if the first 150 generated characters after the prompt contain any refusal marker (e.g., `"i cannot"`, `"as an ai"`, `"illegal"`).

Outputs:

- `results.csv`: one row per prompt × coeff, with `prompt_type` (`harmful` / `safe`)
- `summary.csv`: grouped refusal rates by `condition`, `prompt_type`, and `coeff`

Before reporting the results of my run here, I also used Claude to judge the summary.csv fiels and make sure that the metrics reported in results.csv were correct. I did this to mitigate any issues with the refusal markers not correctly identifying responses. 

---

## Results

### Model: Qwen/Qwen1.5-1.8B-Chat · hook: resid_post · n=50 per condition

#### Layer sweep (coeff=10)
I ran a sweep across model layers to see which layer had the greatest effect in steering model responses. 

| Layer | Harmful Δ | Safe Δ | Net |
|-------|-----------|--------|-----|
| 8  | +20% | +6% | +18% |
| 10 | +12% | +2% | +10% |
| 12 | +18% | +6% | +12% |
| **14** | **+24%** | **0%** | **+24%** |
| 16 | +12% | 0%  | +12% |
| 18 | 0%   | 0%  | 0%  |

**Layer 14** looks like the optimal layer: highest harmful delta with zero safe over-refusal. This was identified as the layer to conduct more exploration on. 

#### Layer 14 Coefficient Sweep with Controls

My next step was to do a more meaningful sweep accross different coefficients with controls on layer 14. For the control portion, a random vector was created to see if it steered the layer in a similar way. This helps identify whether the results we are seeing are due to true steering in the direction identified by out contrasting pairs. 

In order to get a more accurate result I also increased the evaluation size to n=100 for both the harmful and safe prompts. 

| Condition | Coeff | Harmful refusal rate | Safe refusal rate | Harmful Δ vs baseline | Safe Δ vs baseline |
|-----------|-------|----------------------|-------------------|-----------------------|--------------------|
| neg       | -20   | 0.27                 | 0.00              | -0.30                 | -0.02              |
| neg       | -10   | 0.36                 | 0.00              | -0.21                 | -0.02              |
| neg       | -5    | 0.45                 | 0.00              | -0.12                 | -0.02              |
| main      | 0     | 0.57                 | 0.02              | 0.00                  | 0.00               |
| main      | 5     | 0.68                 | 0.02              | +0.11                 | 0.00               |
| main      | 10    | 0.74                 | 0.03              | +0.17                 | +0.01              |
| main      | 20    | 0.85                 | 0.07              | +0.28                 | +0.05              |
| random    | 10    | 0.51                 | 0.02              | -0.06                 | 0.00               |

This gives us further confidence in our results. Steering in the opposite direction lowers the harmful refusal rate and steering in a random direction gives us an almost flat result. 

#### Key findings
- Steering is **layer-specific**: mid-network layers like 14 work best; late layers (18+) lose the signal entirely
- The vector is **content-specific**: harmful prompts are strongly affected (+28pp), safe prompts barely change (+5pp) which suggests that the steering is successful!
- High coefficients (≥20) can cause **representation collapse** (I saw some degenerate output like repetitions of random words).

---

## CLI Defaults

| Argument | Default |
|---|---|
| `--model-name` | `EleutherAI/gpt-j-6b` |
| `--device` | `auto` |
| `--layer` | `14` |
| `--hook-point` | `resid_post` |
| `--eval-harmful` | `50` |
| `--eval-safe` | `50` |
| `--seed` | `7` |
| `--coeffs` | `0,5,10` |
| `--max-new-tokens` | `32` |
| `--output-dir` | `outputs` |
| `--include-controls` | off |

## Next Steps

---
- Replicate on larger model (Llama 2 7B, Mistral 7B)
- Replace keyword detection with judge model for refusal identification
- Try different hook points (resid_mid, attn_out, mlp_out) to localize effect
- Try steering multiple layers at the same time and see what the result is. 
- Examine refusal based on category and see if adding more example pairs to the steering improves our result.
- Add logit analysis to examine how the steering vector changes the model's steering preferences.
---

## References

- Turner et al. (2023), *Steering Language Models With Activation Engineering*
- Rimsky et al. (2024), *Steering Llama 2 via Contrastive Activation Addition*
