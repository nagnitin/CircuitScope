"""
experiments/11_cross_task_patching.py
======================================
Part 4: Novel Extension — Cross-Task Causal Activation Patching

Tests the causal transfer of Name Mover attention heads between the IOI task
and the Pronoun Resolution task.

Ablation/patching pipeline:
  1. Direction A (Pronoun -> Corrupted IOI):
     Take a corrupted IOI prompt, patch clean activations from a matching Pronoun
     resolution prompt at target Name Mover head locations, and measure IOI logit_diff recovery.

  2. Direction B (IOI -> Corrupted Pronoun):
     Take a corrupted Pronoun prompt, patch clean activations from a matching IOI
     prompt at target Name Mover head locations, and measure Pronoun logit_diff recovery.

  3. Same-Task Control Baseline (IOI -> Corrupted IOI):
     Patch clean IOI activations into corrupted IOI runs as the reference baseline.

Produces:
  outputs/11_cross_task_patching/results/cross_task_patching.csv
  outputs/11_cross_task_patching/results/cross_task_summary.json
  outputs/11_cross_task_patching/results/experiment_metadata.json
  outputs/11_cross_task_patching/figures/31_cross_task_patching_bars.png
"""

from __future__ import annotations
import argparse, sys, time, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.logger import get_logger, silence_external_loggers
from src.utils.reproducibility import set_seed
from src.utils.io_utils import ensure_dirs, save_csv, save_json, save_figure
from src.model.loader import load_model
from src.data.ioi_dataset import IOIDataset
from src.data.pronoun_dataset import PronounDataset
from src.evaluation.metrics import compute_logit_diff
from src.analysis.statistics import bootstrap_ci


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Task Activation Patching")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--n-samples", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("circuitscope.cross_task_patching", level=log_level,
                        log_dir=config["paths"]["logs_dir"])
    silence_external_loggers()

    paths = config["paths"]
    paths["figures_dir"] = paths["outputs_dir"] + "/11_cross_task_patching/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/11_cross_task_patching/results"
    ensure_dirs(paths["figures_dir"], paths["results_dir"], paths["logs_dir"])
    set_seed(config.get("seed", 42))

    # ── Load Model & Datasets ─────────────────────────────────────────────
    logger.info("Loading model…")
    model = load_model(config["model"]["name"], device=config["model"]["device"])
    device = next(model.parameters()).device

    logger.info(f"Generating IOI and Pronoun datasets ({args.n_samples} samples each)…")
    dataset_ioi = IOIDataset(model, n_prompts=args.n_samples, seed=42).generate()
    dataset_pronoun = PronounDataset(model, n_prompts=args.n_samples, seed=42).generate()

    n = min(args.n_samples, len(dataset_ioi), len(dataset_pronoun))

    # Target heads to test
    target_heads = [
        ("L8H6", 8, 6, "Name Mover"),
        ("L8H10", 8, 10, "Name Mover"),
        ("L5H5", 5, 5, "Name Mover"),
        ("L7H9", 7, 9, "Name Mover"),
        ("L10H0", 10, 0, "Helper"),
        ("L9H8", 9, 8, "Helper"),
        ("L9H6", 9, 6, "Helper"),
        ("L0H0", 0, 0, "Neutral Control"),
        ("L2H2", 2, 2, "Neutral Control"),
    ]

    # ── Tokenize Prompts ──────────────────────────────────────────────────
    clean_ioi_prompts = dataset_ioi.get_clean_prompts()[:n]
    corrupted_ioi_prompts = dataset_ioi.get_corrupted_prompts()[:n]
    ioi_io = dataset_ioi.get_io_token_ids()[:n]
    ioi_s = dataset_ioi.get_s_token_ids()[:n]
    ioi_distractor = [dataset_ioi._get_token_id(p.distractor_name) for p in dataset_ioi.prompts[:n]]

    clean_pronoun_prompts = dataset_pronoun.get_clean_prompts()[:n]
    corrupted_pronoun_prompts = dataset_pronoun.get_corrupted_prompts()[:n]
    pronoun_target = dataset_pronoun.get_io_token_ids()[:n]
    pronoun_s = dataset_pronoun.get_s_token_ids()[:n]
    pronoun_distractor = [dataset_pronoun._get_token_id(p.speaker_name) for p in dataset_pronoun.prompts[:n]]

    # Helper function to run model on prompts in batches
    def run_all_in_batches(prompts, h_names, b_size=args.batch_size):
        all_logits_list = []
        all_tokens_list = []
        all_lens_list = []
        cache_dict_list = {h: [] for h in h_names}

        for i in range(0, len(prompts), b_size):
            batch_p = prompts[i:i + b_size]
            token_lists = [model.to_tokens(p, prepend_bos=True)[0].tolist() for p in batch_p]
            max_len = max(len(t) for t in token_lists)
            bos_id = model.tokenizer.bos_token_id
            padded = [t + [bos_id] * (max_len - len(t)) for t in token_lists]
            tokens = torch.tensor(padded, dtype=torch.long, device=device)
            seq_lens = [len(t) for t in token_lists]

            logits, cache = model.run_with_cache(tokens, names_filter=h_names)
            all_logits_list.append(logits.detach())
            all_tokens_list.append(tokens)
            all_lens_list.extend(seq_lens)
            for h in h_names:
                cache_dict_list[h].append(cache[h].detach())

        # Concatenate tokens, logits, and cache along batch dimension
        max_batch_seq = max(t.shape[1] for t in all_tokens_list)
        padded_tokens = []
        padded_logits = []
        bos_id = model.tokenizer.bos_token_id
        for t, l in zip(all_tokens_list, all_logits_list):
            if t.shape[1] < max_batch_seq:
                pad_t = torch.full((t.shape[0], max_batch_seq - t.shape[1]), bos_id, dtype=torch.long, device=device)
                t = torch.cat([t, pad_t], dim=1)
                pad_l = torch.zeros((l.shape[0], max_batch_seq - l.shape[1], l.shape[2]), dtype=l.dtype, device=device)
                l = torch.cat([l, pad_l], dim=1)
            padded_tokens.append(t)
            padded_logits.append(l)

        concat_tokens = torch.cat(padded_tokens, dim=0)
        concat_logits = torch.cat(padded_logits, dim=0)

        concat_cache = {}
        for h in h_names:
            padded_h = []
            for c_tensor in cache_dict_list[h]:
                if c_tensor.shape[1] < max_batch_seq:
                    pad_c = torch.zeros((c_tensor.shape[0], max_batch_seq - c_tensor.shape[1], c_tensor.shape[2], c_tensor.shape[3]), dtype=c_tensor.dtype, device=device)
                    c_tensor = torch.cat([c_tensor, pad_c], dim=1)
                padded_h.append(c_tensor)
            concat_cache[h] = torch.cat(padded_h, dim=0)

        return concat_tokens, all_lens_list, concat_logits, concat_cache

    hook_names = list(set([f"blocks.{l}.attn.hook_z" for _, l, _, _ in target_heads]))

    logger.info("Computing baselines for IOI and Pronoun tasks…")
    _, ioi_clean_lens, ioi_clean_logits, ioi_clean_cache = run_all_in_batches(clean_ioi_prompts, hook_names)
    corr_tokens_ioi, ioi_corr_lens, ioi_corr_logits, _ = run_all_in_batches(corrupted_ioi_prompts, hook_names)

    _, pron_clean_lens, pron_clean_logits, pron_clean_cache = run_all_in_batches(clean_pronoun_prompts, hook_names)
    corr_tokens_pron, pron_corr_lens, pron_corr_logits, _ = run_all_in_batches(corrupted_pronoun_prompts, hook_names)

    ioi_clean_lds = [compute_logit_diff(ioi_clean_logits[i, ioi_clean_lens[i]-1, :], ioi_io[i], ioi_s[i]) for i in range(n)]
    ioi_corr_lds  = [compute_logit_diff(ioi_corr_logits[i, ioi_corr_lens[i]-1, :], ioi_io[i], ioi_distractor[i]) for i in range(n)]

    pron_clean_lds = [compute_logit_diff(pron_clean_logits[i, pron_clean_lens[i]-1, :], pronoun_target[i], pronoun_s[i]) for i in range(n)]
    pron_corr_lds  = [compute_logit_diff(pron_corr_logits[i, pron_corr_lens[i]-1, :], pronoun_target[i], pronoun_distractor[i]) for i in range(n)]

    mean_ioi_clean = float(np.mean(ioi_clean_lds))
    mean_ioi_corr  = float(np.mean(ioi_corr_lds))
    mean_pron_clean = float(np.mean(pron_clean_lds))
    mean_pron_corr  = float(np.mean(pron_corr_lds))

    logger.info(f"IOI Clean LD: {mean_ioi_clean:+.4f}, Corrupted LD: {mean_ioi_corr:+.4f}")
    logger.info(f"Pronoun Clean LD: {mean_pron_clean:+.4f}, Corrupted LD: {mean_pron_corr:+.4f}")

    results = []

    def run_patched_in_batches(tokens, hook_z_name, donor_z, target_head, b_size=args.batch_size):
        all_patched_logits = []
        for i in range(0, tokens.shape[0], b_size):
            batch_tokens = tokens[i:i + b_size]
            batch_donor = donor_z[i:i + b_size]
            def hook_fn(value, hook):
                v = value.clone()
                min_seq = min(v.shape[1], batch_donor.shape[1])
                v[:, :min_seq, target_head, :] = batch_donor[:, :min_seq, target_head, :].to(v.device)
                return v
            logits = model.run_with_hooks(batch_tokens, fwd_hooks=[(hook_z_name, hook_fn)])
            all_patched_logits.append(logits.detach())
        return torch.cat(all_patched_logits, dim=0)

    for head_label, layer, head, head_type in target_heads:
        hook_z_name = f"blocks.{layer}.attn.hook_z"

        # ── 1. Same-Task Control: Clean IOI -> Corrupted IOI ───────────────
        ioi_clean_z = ioi_clean_cache[hook_z_name] # [n, seq, n_heads, d_head]

        patched_logits_same = run_patched_in_batches(corr_tokens_ioi, hook_z_name, ioi_clean_z, head)
        patched_lds_same = [compute_logit_diff(patched_logits_same[i, ioi_corr_lens[i]-1, :], ioi_io[i], ioi_distractor[i]) for i in range(n)]
        rec_same = [(patched_lds_same[i] - ioi_corr_lds[i]) / (ioi_clean_lds[i] - ioi_corr_lds[i] + 1e-8) for i in range(n)]
        score_same = float(np.mean(rec_same))
        _, ci_lo_same, ci_hi_same = bootstrap_ci(np.array(rec_same), seed=42)

        # ── 2. Direction A: Pronoun -> Corrupted IOI ────────────────────────
        pron_clean_z = pron_clean_cache[hook_z_name]

        patched_logits_p2i = run_patched_in_batches(corr_tokens_ioi, hook_z_name, pron_clean_z, head)
        patched_lds_p2i = [compute_logit_diff(patched_logits_p2i[i, ioi_corr_lens[i]-1, :], ioi_io[i], ioi_distractor[i]) for i in range(n)]
        rec_p2i = [(patched_lds_p2i[i] - ioi_corr_lds[i]) / (ioi_clean_lds[i] - ioi_corr_lds[i] + 1e-8) for i in range(n)]
        score_p2i = float(np.mean(rec_p2i))
        _, ci_lo_p2i, ci_hi_p2i = bootstrap_ci(np.array(rec_p2i), seed=42)

        # ── 3. Direction B: IOI -> Corrupted Pronoun ────────────────────────
        patched_logits_i2p = run_patched_in_batches(corr_tokens_pron, hook_z_name, ioi_clean_z, head)
        patched_lds_i2p = [compute_logit_diff(patched_logits_i2p[i, pron_corr_lens[i]-1, :], pronoun_target[i], pronoun_distractor[i]) for i in range(n)]
        rec_i2p = [(patched_lds_i2p[i] - pron_corr_lds[i]) / (pron_clean_lds[i] - pron_corr_lds[i] + 1e-8) for i in range(n)]
        score_i2p = float(np.mean(rec_i2p))
        _, ci_lo_i2p, ci_hi_i2p = bootstrap_ci(np.array(rec_i2p), seed=42)

        results.append({
            "head_label": head_label,
            "layer": layer,
            "head": head,
            "head_type": head_type,
            "same_task_recovery": round(score_same, 4),
            "same_task_ci_lo": round(ci_lo_same, 4),
            "same_task_ci_hi": round(ci_hi_same, 4),
            "pronoun_to_ioi_recovery": round(score_p2i, 4),
            "pronoun_to_ioi_ci_lo": round(ci_lo_p2i, 4),
            "pronoun_to_ioi_ci_hi": round(ci_hi_p2i, 4),
            "ioi_to_pronoun_recovery": round(score_i2p, 4),
            "ioi_to_pronoun_ci_lo": round(ci_lo_i2p, 4),
            "ioi_to_pronoun_ci_hi": round(ci_hi_i2p, 4),
            "mean_cross_task_recovery": round((score_p2i + score_i2p) / 2.0, 4),
        })

        logger.info(
            f"Head {head_label} ({head_type}): "
            f"Same-Task={score_same:+.3f}, "
            f"Pronoun->IOI={score_p2i:+.3f}, "
            f"IOI->Pronoun={score_i2p:+.3f}"
        )

    results_df = pd.DataFrame(results)
    save_csv(results_df, paths["results_dir"] + "/cross_task_patching.csv")

    nm_df = results_df[results_df["head_type"]=="Name Mover"]
    neu_df = results_df[results_df["head_type"]=="Neutral Control"]

    summary = {
        "n_samples": n,
        "ioi_clean_ld": round(mean_ioi_clean, 4),
        "ioi_corrupted_ld": round(mean_ioi_corr, 4),
        "pronoun_clean_ld": round(mean_pron_clean, 4),
        "pronoun_corrupted_ld": round(mean_pron_corr, 4),
        "name_mover_same_task_recovery": round(float(nm_df["same_task_recovery"].mean()), 4),
        "name_mover_cross_recovery": round(float(nm_df["mean_cross_task_recovery"].mean()), 4),
        "neutral_cross_recovery": round(float(neu_df["mean_cross_task_recovery"].mean()), 4),
        "causal_transfer_verdict": "PARTIAL_FUNCTIONAL_OVERLAP" if float(nm_df["mean_cross_task_recovery"].mean()) > float(neu_df["mean_cross_task_recovery"].mean()) else "NO_TRANSFER"
    }
    save_json(summary, paths["results_dir"] + "/cross_task_summary.json")

    # ── Figure 31: Cross-Task Patching Recovery Bar Chart ────────────────
    plot_df = []
    for _, row in results_df.iterrows():
        plot_df.append({"Head": row["head_label"], "Direction": "Same-Task (IOI -> IOI)", "Recovery Score": row["same_task_recovery"], "Head Type": row["head_type"]})
        plot_df.append({"Head": row["head_label"], "Direction": "Cross-Task (Pronoun -> IOI)", "Recovery Score": row["pronoun_to_ioi_recovery"], "Head Type": row["head_type"]})
        plot_df.append({"Head": row["head_label"], "Direction": "Cross-Task (IOI -> Pronoun)", "Recovery Score": row["ioi_to_pronoun_recovery"], "Head Type": row["head_type"]})
    p_df = pd.DataFrame(plot_df)

    fig = px.bar(
        p_df, x="Head", y="Recovery Score", color="Direction", barmode="group",
        title="Cross-Task Activation Patching: Causal Recovery at Target Heads",
        template="plotly_dark",
    )
    save_figure(fig, paths["figures_dir"] + "/31_cross_task_patching_bars", formats=["png", "html"])

    print("\n--- Cross-Task Activation Patching Results ----------------------")
    print(results_df[["head_label", "head_type", "same_task_recovery", "pronoun_to_ioi_recovery", "ioi_to_pronoun_recovery", "mean_cross_task_recovery"]].to_string(index=False))
    print(f"\nMean Same-Task Recovery (Name Movers) : {summary['name_mover_same_task_recovery']:.4f}")
    print(f"Mean Cross-Task Recovery (Name Movers): {summary['name_mover_cross_recovery']:.4f}")
    print(f"Mean Cross-Task Recovery (Neutral)    : {summary['neutral_cross_recovery']:.4f}")
    print(f"Verdict: {summary['causal_transfer_verdict']}")
    print("- Saved outputs to outputs/11_cross_task_patching/")

if __name__ == "__main__":
    main()
