from typing import Iterable, Tuple

import torch
from transformer_lens import HookedTransformer


def load_model(model_name: str, device: str | None = None) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    return model


@torch.inference_mode()
def compute_steering_vector(
    model: HookedTransformer,
    pairs: Iterable[Tuple[str, str]],
    layer: int,
    hook_point: str,
) -> torch.Tensor:
    
    hook_name = f"blocks.{layer}.hook_{hook_point}"
    diffs: list[torch.Tensor] = []

    for pos_text, neg_text in pairs:
        _, pos_cache = model.run_with_cache(pos_text, names_filter=lambda n: n == hook_name)
        _, neg_cache = model.run_with_cache(neg_text, names_filter=lambda n: n == hook_name)

        pos_a = pos_cache[hook_name][0, -1, :].detach()
        neg_a = neg_cache[hook_name][0, -1, :].detach()

        diffs.append(pos_a - neg_a)

    if not diffs:
        raise ValueError("No pairs provided.")

    vec = torch.stack(diffs, dim=0).mean(dim=0)  # [d_model]
    return vec / (vec.norm() + 1e-8)


def make_random_direction(ref: torch.Tensor, seed: int = 0, n_samples: int = 10) -> torch.Tensor:
    ref_cpu = ref.detach().to(device="cpu", dtype=torch.float32)
    ref_flat = ref_cpu.reshape(-1)
    ref_norm_sq = (ref_flat @ ref_flat).clamp_min(1e-8)

    g = torch.Generator(device="cpu")
    vectors = []
    for i in range(n_samples):
        g.manual_seed(seed + i)
        v = torch.randn(ref_cpu.shape, generator=g, device="cpu", dtype=torch.float32)
        proj = (v.reshape(-1) @ ref_flat) / ref_norm_sq
        v = v - proj * ref_cpu
        vectors.append(v / v.norm().clamp_min(1e-8))

    mean_v = torch.stack(vectors).mean(0)
    mean_v = mean_v / mean_v.norm().clamp_min(1e-8)
    return mean_v.to(device=ref.device, dtype=ref.dtype)


@torch.inference_mode()
def generate_with_steering(
    model: HookedTransformer,
    prompt: str,
    vector: torch.Tensor,
    coeff: float,
    layer: int,
    hook_point: str,
    max_new_tokens: int,
) -> str:
    
    hook_name = f"blocks.{layer}.hook_{hook_point}"
    sv = vector.to(device=model.cfg.device, dtype=model.cfg.dtype)

    if sv.ndim == 1:
        sv = sv.view(1, 1, -1)
    elif sv.ndim == 2:
        sv = sv.unsqueeze(0)
    else:
        raise ValueError(f"Expected vector to be 1D or 2D, got shape {tuple(sv.shape)}")


    def add_steering(resid: torch.Tensor, hook) -> torch.Tensor:

        if coeff == 0.0 :
            return resid

        return resid + coeff * sv

    with model.hooks(fwd_hooks=[(hook_name, add_steering)]):
        return model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
            return_type="str",
        )
