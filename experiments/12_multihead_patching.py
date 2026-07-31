"""
experiments/12_multihead_patching.py
======================================
Part 5: Novel Extension — Multi-Head Sub-Circuit Group Patching

Extends experiment 11 (single-head patching) by simultaneously patching
the FULL Name Mover group and the Name Mover + S-Inhibition group,
in both cross-task directions.

This tests whether the NO_TRANSFER verdict from exp 11 survives when the
entire functional sub-circuit is transplanted at once — ruling out the
possibility that single-head tests underestimate transfer due to requiring
group-level coordination.

Groups tested:
  Group A - Name Mover heads:           L8H6, L8H10, L5H5, L7H9
  Group B - Name Mover + S-Inhibition:  + L7H3, L7H9 (S-Inhibition, layers 7-8)

Directions:
  1. Same-Task Control: IOI clean -> IOI corrupted
  2. Direction A: Pronoun clean -> IOI corrupted
  3. Direction B: IOI clean -> Pronoun corrupted

Produces:
  outputs/12_multihead_patching/results/multihead_patching.csv
  outputs/12_multihead_patching/results/multihead_summary.json
  outputs/12_multihead_patching/figures/32_multihead_patching_bars.png

Usage:
  python experiments/12_multihead_patching.py
  python experiments/12_multihead_patching.py --n-samples 150
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

from src.utils.logger import get_logger, silence_external_loggers
from src.utils.reproducibility import set_seed
from src.utils.io_utils import ensure_dirs, save_csv, save_json, save_figure
from src.model.loader import load_model
from src.data.ioi_dataset import IOIDataset
from src.data.pronoun_dataset import PronounDataset
from src.evaluation.metrics import compute_logit_diff
from src.analysis.statistics import bootstrap_ci

# ---------------------------------------------------------------------------
# Head group definitions
# ---------------------------------------------------------------------------
NAME_MOVER_HEADS = [(8, 6), (8, 10), (5, 5), (7, 9)]
S_INHIBITION_HEADS = [(7, 3), (7, 9)]  # Layers 7-8, from Wang et al. canonical spec

GROUPS = [
    ("Name Mover Group (A)",   NAME_MOVER_HEADS),
    ("NM + S-Inhibition (B)", sorted(set(NAME_MOVER_HEADS + S_INHIBITION_HEADS))),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Head Sub-Circuit Group Patching")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--n-samples", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = get_logger("circuitscope.multihead_patching", level=log_level,
                        log_dir=config["paths"]["logs_dir"])
    silence_external_loggers()

    paths = config["paths"]
    paths["figures_dir"] = paths["outputs_dir"] + "/12_multihead_patching/figures"
    paths["results_dir"] = paths["outputs_dir"] + "/12_multihead_patching/results"
    ensure_dirs(paths["figures_dir"], paths["results_dir"], paths["logs_dir"])

    SEED = config.get("seed", 42)
    set_seed(SEED)

    # ---- Load Model & Datasets -----------------------------------------------
    logger.info("Loading model...")
    model = load_model(config["model"]["name"], device=config["model"]["device"])
    device = next(model.parameters()).device

    logger.info(f"Generating IOI and Pronoun datasets ({args.n_samples} samples each)...")
    dataset_ioi = IOIDataset(model, n_prompts=args.n_samples, seed=SEED).generate()
    dataset_pronoun = PronounDataset(model, n_prompts=args.n_samples, seed=SEED).generate()

    n = min(args.n_samples, len(dataset_ioi), len(dataset_pronoun))

    # ---- Collect unique hook names for all groups ----------------------------
    all_layers = set()
    for _, heads in GROUPS:
        for l, _ in heads:
            all_layers.add(l)
    hook_names = [f"blocks.{l}.attn.hook_z" for l in sorted(all_layers)]

    # ---- Tokenize Prompts ----------------------------------------------------
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

    # ---- Batch run helper ----------------------------------------------------
    def run_all_in_batches(prompts, h_names, b_size=args.batch_size):
        all_logits_list, all_tokens_list, all_lens_list = [], [], []
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

        max_batch_seq = max(t.shape[1] for t in all_tokens_list)
        padded_tokens, padded_logits = [], []
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
                    pad_c = torch.zeros(
                        (c_tensor.shape[0], max_batch_seq - c_tensor.shape[1],
                         c_tensor.shape[2], c_tensor.shape[3]),
                        dtype=c_tensor.dtype, device=device,
                    )
                    c_tensor = torch.cat([c_tensor, pad_c], dim=1)
                padded_h.append(c_tensor)
            concat_cache[h] = torch.cat(padded_h, dim=0)

        return concat_tokens, all_lens_list, concat_logits, concat_cache

    logger.info("Computing baselines for IOI and Pronoun tasks...")
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

    # ---- Multi-head joint patching function ----------------------------------
    def run_multihead_patching(recipient_tokens, recipient_lens,
                               recipient_clean_lds, recipient_corr_lds,
                               recipient_target_ids, recipient_foil_ids,
                               donor_cache, group_heads, b_size=args.batch_size):
        """
        Simultaneously patch all heads in group_heads from donor_cache into
        corrupted recipient_tokens, return per-prompt recovery scores.
        """
        # Build grouped hooks: group by layer so we register one hook per layer
        from collections import defaultdict
        layer_heads_map = defaultdict(list)
        for l, h in group_heads:
            layer_heads_map[l].append(h)

        all_patched_logits = []
        for i in range(0, recipient_tokens.shape[0], b_size):
            batch_tokens  = recipient_tokens[i:i + b_size]
            batch_sz = batch_tokens.shape[0]

            fwd_hooks = []
            for layer, heads_in_layer in layer_heads_map.items():
                hook_name = f"blocks.{layer}.attn.hook_z"
                batch_donor = donor_cache[hook_name][i:i + batch_sz]

                def make_hook(batch_donor_z, heads_list):
                    def hook_fn(value, hook):
                        v = value.clone()
                        min_seq = min(v.shape[1], batch_donor_z.shape[1])
                        for h_idx in heads_list:
                            v[:, :min_seq, h_idx, :] = batch_donor_z[:, :min_seq, h_idx, :].to(v.device)
                        return v
                    return hook_fn

                fwd_hooks.append((hook_name, make_hook(batch_donor, heads_in_layer)))

            logits = model.run_with_hooks(batch_tokens, fwd_hooks=fwd_hooks)
            all_patched_logits.append(logits.detach())

        patched_logits = torch.cat(all_patched_logits, dim=0)

        patched_lds = [
            compute_logit_diff(patched_logits[i, recipient_lens[i]-1, :],
                               recipient_target_ids[i], recipient_foil_ids[i])
            for i in range(len(recipient_lens))
        ]
        recoveries = [
            (patched_lds[i] - recipient_corr_lds[i]) / (recipient_clean_lds[i] - recipient_corr_lds[i] + 1e-8)
            for i in range(len(recipient_lens))
        ]
        return recoveries

    # ---- Run all groups ------------------------------------------------------
    results = []
    for group_label, group_heads in GROUPS:
        logger.info(f"Patching group: {group_label} | Heads: {group_heads}")

        # 1. Same-Task Control: IOI clean -> IOI corrupted
        rec_same = run_multihead_patching(
            corr_tokens_ioi, ioi_corr_lens,
            ioi_clean_lds, ioi_corr_lds,
            ioi_io, ioi_distractor,
            ioi_clean_cache, group_heads,
        )
        score_same = float(np.mean(rec_same))
        _, ci_lo_same, ci_hi_same = bootstrap_ci(np.array(rec_same), seed=SEED)

        # 2. Direction A: Pronoun clean -> IOI corrupted
        rec_p2i = run_multihead_patching(
            corr_tokens_ioi, ioi_corr_lens,
            ioi_clean_lds, ioi_corr_lds,
            ioi_io, ioi_distractor,
            pron_clean_cache, group_heads,
        )
        score_p2i = float(np.mean(rec_p2i))
        _, ci_lo_p2i, ci_hi_p2i = bootstrap_ci(np.array(rec_p2i), seed=SEED)

        # 3. Direction B: IOI clean -> Pronoun corrupted
        rec_i2p = run_multihead_patching(
            corr_tokens_pron, pron_corr_lens,
            pron_clean_lds, pron_corr_lds,
            pronoun_target, pronoun_distractor,
            ioi_clean_cache, group_heads,
        )
        score_i2p = float(np.mean(rec_i2p))
        _, ci_lo_i2p, ci_hi_i2p = bootstrap_ci(np.array(rec_i2p), seed=SEED)

        mean_cross = round((score_p2i + score_i2p) / 2.0, 4)

        results.append({
            "group_label": group_label,
            "n_heads_patched": len(group_heads),
            "heads": str(group_heads),
            "same_task_recovery": round(score_same, 4),
            "same_task_ci_lo": round(ci_lo_same, 4),
            "same_task_ci_hi": round(ci_hi_same, 4),
            "pronoun_to_ioi_recovery": round(score_p2i, 4),
            "pronoun_to_ioi_ci_lo": round(ci_lo_p2i, 4),
            "pronoun_to_ioi_ci_hi": round(ci_hi_p2i, 4),
            "ioi_to_pronoun_recovery": round(score_i2p, 4),
            "ioi_to_pronoun_ci_lo": round(ci_lo_i2p, 4),
            "ioi_to_pronoun_ci_hi": round(ci_hi_i2p, 4),
            "mean_cross_task_recovery": mean_cross,
        })

        logger.info(
            f"  {group_label}: Same-Task={score_same:+.4f}, "
            f"Pronoun->IOI={score_p2i:+.4f} [{ci_lo_p2i:+.3f},{ci_hi_p2i:+.3f}], "
            f"IOI->Pronoun={score_i2p:+.4f} [{ci_lo_i2p:+.3f},{ci_hi_i2p:+.3f}], "
            f"Mean-Cross={mean_cross:+.4f}"
        )

    results_df = pd.DataFrame(results)
    save_csv(results_df, paths["results_dir"] + "/multihead_patching.csv")

    # ---- Summary JSON --------------------------------------------------------
    group_a = results_df[results_df["group_label"].str.startswith("Name Mover Group")].iloc[0]
    group_b = results_df[results_df["group_label"].str.startswith("NM + S-Inhibition")].iloc[0]

    # Verdict: group-level cross-task recovery vs single-head baseline
    single_head_cross = -0.0597  # from exp 11 cross_task_summary.json
    group_a_cross = float(group_a["mean_cross_task_recovery"])
    group_b_cross = float(group_b["mean_cross_task_recovery"])

    TRANSFER_THRESHOLD = 0.05  # recovery > 5% of the clean-corrupted gap = meaningful transfer

    if group_b_cross > TRANSFER_THRESHOLD:
        verdict = "GROUP_LEVEL_TRANSFER"
    elif group_a_cross > TRANSFER_THRESHOLD:
        verdict = "NAME_MOVER_GROUP_TRANSFER_ONLY"
    else:
        verdict = "NO_TRANSFER_EVEN_AT_GROUP_LEVEL"

    summary = {
        "n_samples": n,
        "random_seed": SEED,
        "ioi_clean_ld": round(mean_ioi_clean, 4),
        "ioi_corrupted_ld": round(mean_ioi_corr, 4),
        "pronoun_clean_ld": round(mean_pron_clean, 4),
        "pronoun_corrupted_ld": round(mean_pron_corr, 4),
        "exp11_single_head_cross_recovery": single_head_cross,
        "group_a_name_mover_same_task_recovery": round(float(group_a["same_task_recovery"]), 4),
        "group_a_name_mover_cross_recovery": round(group_a_cross, 4),
        "group_a_pronoun_to_ioi_recovery": round(float(group_a["pronoun_to_ioi_recovery"]), 4),
        "group_a_ioi_to_pronoun_recovery": round(float(group_a["ioi_to_pronoun_recovery"]), 4),
        "group_b_nm_si_same_task_recovery": round(float(group_b["same_task_recovery"]), 4),
        "group_b_nm_si_cross_recovery": round(group_b_cross, 4),
        "group_b_pronoun_to_ioi_recovery": round(float(group_b["pronoun_to_ioi_recovery"]), 4),
        "group_b_ioi_to_pronoun_recovery": round(float(group_b["ioi_to_pronoun_recovery"]), 4),
        "causal_transfer_verdict": verdict,
        "transfer_threshold_used": TRANSFER_THRESHOLD,
    }
    save_json(summary, paths["results_dir"] + "/multihead_summary.json")

    # ---- Figure 32: Group Patching Bar Chart ---------------------------------
    plot_rows = []
    for _, row in results_df.iterrows():
        plot_rows += [
            {"Group": row["group_label"], "Direction": "Same-Task (IOI->IOI)", "Recovery": row["same_task_recovery"]},
            {"Group": row["group_label"], "Direction": "Cross-Task (Pronoun->IOI)", "Recovery": row["pronoun_to_ioi_recovery"]},
            {"Group": row["group_label"], "Direction": "Cross-Task (IOI->Pronoun)", "Recovery": row["ioi_to_pronoun_recovery"]},
        ]
    plot_df = pd.DataFrame(plot_rows)

    fig = px.bar(
        plot_df, x="Group", y="Recovery", color="Direction", barmode="group",
        title="Multi-Head Group Patching: Cross-Task Causal Recovery",
        template="plotly_dark",
    )
    save_figure(fig, paths["figures_dir"] + "/32_multihead_patching_bars", formats=["png", "html"])

    # ---- Console Summary -----------------------------------------------------
    print("\n--- Multi-Head Sub-Circuit Patching Results -------------------")
    print(results_df[["group_label", "n_heads_patched", "same_task_recovery",
                       "pronoun_to_ioi_recovery", "ioi_to_pronoun_recovery",
                       "mean_cross_task_recovery"]].to_string(index=False))
    print(f"\nSingle-Head Baseline (Exp 11): cross_recovery = {single_head_cross:+.4f}")
    print(f"Group A (Name Movers only) :  cross_recovery = {group_a_cross:+.4f}")
    print(f"Group B (NM + S-Inhibition):  cross_recovery = {group_b_cross:+.4f}")
    print(f"\nVerdict: {verdict}")
    print("- Saved outputs to outputs/12_multihead_patching/")


if __name__ == "__main__":
    main()
