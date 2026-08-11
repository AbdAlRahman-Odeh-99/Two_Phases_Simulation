# -*- coding: utf-8 -*-
"""
inspect_marginal_gains.py -- standalone diagnostic for the submodular
objective used by greedy_oracle / greedy_chain.

Run it from the repo root:

    python3 inspect_marginal_gains.py
    python3 inspect_marginal_gains.py --nviews 6 --nclasses 3 --seed 1
    python3 inspect_marginal_gains.py --budget 0.5 --lam 0.05
"""

from __future__ import annotations

import argparse
import re
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# The objective, inlined verbatim from core/submodular_greedy.py so this
# script is readable on its own and runs without numba.
# ─────────────────────────────────────────────────────────────────────────
def bhattacharyya_error_rate(diff_mean_sq_mat):
    d_norm = np.sum(diff_mean_sq_mat, axis=0)          # (nc, nc)
    acc = np.exp(-0.125 * d_norm)
    return np.maximum(1.0 - acc, 0.0)


def multiclass_risk(diff_mean_sq_mat):
    return 0.5 * np.sum(bhattacharyya_error_rate(diff_mean_sq_mat))


def pairwise_diff_sq_from_means(est_means):
    mean_tr = np.asarray(est_means, dtype=np.float64).T   # (nviews, nc)
    w_diff = mean_tr[:, :, None] - mean_tr[:, None, :]
    return np.square(w_diff)                              # (nviews, nc, nc)


def f_raw(diff_sq, idx):
    """The objective as written in the notebook: multiclass_risk(S)."""
    if len(idx) == 0:
        return 0.0
    return float(multiclass_risk(diff_sq[np.array(sorted(idx))]))


def f_flipped(diff_sq, idx):
    """The objective as currently written in the repo: 1 - multiclass_risk(S)."""
    return 1.0 - f_raw(diff_sq, idx)


# ─────────────────────────────────────────────────────────────────────────
# Local greedy_oracle, structurally identical to the repo's but taking the
# objective as a parameter and optionally narrating each decision.
# ─────────────────────────────────────────────────────────────────────────
def greedy_oracle_local(diff_sq, costs, omd_lambda, remain_budget,
                        free_indices, F, force_free=True, trace=False):
    nview = len(costs)
    current_cost = 0.0
    current_objective = 0.0
    sel_set = []
    avail = set(range(nview))

    for i in list(free_indices):
        mg = F(diff_sq, sel_set + [i]) - current_objective
        if force_free or mg > 0:
            sel_set.append(i)
            current_objective += mg
            avail.remove(i)
    if trace:
        print(f"    after free views: S={sel_set}  objective={current_objective:+.4f}")

    copy_set = list(sel_set)
    step = 0
    while avail and current_cost < remain_budget:
        step += 1
        best_margin, best_add = -1, None
        rows = []
        for i in sorted(avail):
            ci = costs[i]
            if current_cost + ci <= remain_budget:
                mg = F(diff_sq, sel_set + [i]) - current_objective
                ratio = mg / (ci + 1e-9)
                ok = ratio > omd_lambda
                rows.append((i, ci, mg, ratio, ok))
                if ok and ratio > best_margin:
                    best_margin, best_add = ratio, i
        if trace:
            print(f"    step {step}: testing additions to S={sel_set}")
            print(f"      {'view':>5}{'cost':>9}{'margin':>12}{'gain/cost':>13}"
                  f"{'> lambda?':>12}")
            for i, ci, mg, ratio, ok in rows:
                print(f"      {i:>5}{ci:>9.4f}{mg:>12.4f}{ratio:>13.4f}"
                      f"{('YES' if ok else 'no'):>12}")
            print(f"      -> best_add = {best_add}")
        if best_add is not None:
            sel_set.append(best_add)
            current_cost += costs[best_add]
            current_objective = F(diff_sq, sel_set)
            avail.remove(best_add)
        else:
            if trace:
                print("      -> no view beats the shadow price; loop BREAKS here")
            break

    greedy_reward = current_objective
    greedy_idx = list(sel_set)
    best_single, best_giant = None, -np.inf
    for i in range(nview):
        if i in copy_set:
            continue
        if 0 < costs[i] <= remain_budget:
            r = F(diff_sq, copy_set + [i])
            if r > best_giant:
                best_giant, best_single = r, i
    if trace:
        print(f"    giant-item check: best single add = {best_single}, "
              f"value {best_giant:+.4f} vs greedy {greedy_reward:+.4f} "
              f"-> {'TAKEN' if (best_single is not None and best_giant > greedy_reward) else 'rejected'}")

    if best_single is not None and best_giant > greedy_reward:
        final = list(copy_set) + [best_single]
    else:
        final = greedy_idx

    mask = np.array([i in final for i in range(nview)], dtype=bool)
    if not mask.any():
        mask[int(np.argmin(costs))] = True
    return mask


def greedy_chain_local(diff_sq, costs, free_indices, F, force_free=True):
    """Returns the ORDER paid views are appended -- that ordering is the
    whole content of the chain action space."""
    nviews = len(costs)
    sel, objective = [], 0.0
    for i in list(free_indices):
        gain = F(diff_sq, sel + [i]) - objective
        if force_free or gain > 0:
            sel.append(i)
            objective += gain
    if not sel:
        sel = [int(np.argmin(costs))]
        objective = F(diff_sq, sel)

    order, remaining = [], [i for i in range(nviews) if i not in sel]
    while remaining:
        best_ratio, best_add = -np.inf, None
        for i in remaining:
            gain = F(diff_sq, sel + [i]) - objective
            ratio = gain / (costs[i] + 1e-9)
            if ratio > best_ratio:
                best_ratio, best_add = ratio, i
        sel.append(best_add)
        objective = F(diff_sq, sel)
        remaining.remove(best_add)
        order.append(best_add)
    return order


# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nviews", type=int, default=5)
    ap.add_argument("--nclasses", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=2.0,
                    help="mean scale; matches the synthetic generator's default")
    ap.add_argument("--budget", type=float, default=0.6,
                    help="remaining budget for the oracle call (costs sum to 1)")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="omd_lambda / shadow price. Note it is clamped to >= 0 "
                         "everywhere in the repo, so negative values are unreachable.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    means = rng.random(size=(args.nclasses, args.nviews)) * args.scale
    diff_sq = pairwise_diff_sq_from_means(means)

    # view 0 free, the rest paid; normalized so sum(costs) == 1, as the runners do
    raw = np.concatenate([[0.0], rng.random(args.nviews - 1) + 0.5])
    costs = raw / raw.sum()
    free_indices = [i for i in range(args.nviews) if costs[i] == 0]

    print("=" * 78)
    print(f"nviews={args.nviews}  nclasses={args.nclasses}  seed={args.seed}  "
          f"budget={args.budget}  lambda={args.lam}")
    print("costs:", np.array2string(costs, precision=4))
    print("=" * 78)

    # ── A ──
    print("\n[A] f(S) = multiclass_risk(S), accumulating views 0,1,2,...")
    print(f"    {'S':<22}{'f(S)':>10}{'1 - f(S)':>12}")
    S = []
    for i in range(args.nviews):
        S = S + [i]
        fr = f_raw(diff_sq, S)
        print(f"    {str(S):<22}{fr:>10.4f}{1 - fr:>12.4f}")
    print(f"    f(empty) = {f_raw(diff_sq, []):.4f}   "
          f"upper bound nc(nc-1)/2 = {args.nclasses * (args.nclasses - 1) / 2:.1f}")

    # ── B ──
    print("\n[B] marginal gain of adding each view to S = {free views}")
    base = list(free_indices)
    b_raw = f_raw(diff_sq, base)
    b_flip = f_flipped(diff_sq, base)
    print(f"    base S={base}   f(S)={b_raw:.4f}   1-f(S)={b_flip:.4f}")
    print(f"    {'view':>5}{'cost':>9}{'margin (raw)':>15}{'margin (1-f)':>15}"
          f"{'ratio (raw)':>14}{'ratio (1-f)':>14}")
    for i in range(args.nviews):
        if i in base:
            continue
        m_raw = f_raw(diff_sq, base + [i]) - b_raw
        m_flip = f_flipped(diff_sq, base + [i]) - b_flip
        print(f"    {i:>5}{costs[i]:>9.4f}{m_raw:>15.4f}{m_flip:>15.4f}"
              f"{m_raw / (costs[i] + 1e-9):>14.4f}{m_flip / (costs[i] + 1e-9):>14.4f}")
    # ── C ──
    print("\n[C] greedy_oracle trace, CURRENT convention  f(S) := 1 - multiclass_risk(S)")
    m_flip = greedy_oracle_local(diff_sq, costs, args.lam, args.budget,
                                 free_indices, f_flipped, trace=True)
    print(f"    RESULT mask={m_flip.astype(int)}  views={int(m_flip.sum())}  "
          f"cost={costs[m_flip].sum():.4f} of {args.budget}")

    print("\n    greedy_oracle trace, ORIGINAL convention  f(S) := multiclass_risk(S)")
    m_raw = greedy_oracle_local(diff_sq, costs, args.lam, args.budget,
                                free_indices, f_raw, trace=True)
    print(f"    RESULT mask={m_raw.astype(int)}  views={int(m_raw.sum())}  "
          f"cost={costs[m_raw].sum():.4f} of {args.budget}")

    # ── D / E ── the repo's real functions, if importable
    print("\n[D/E] the repo's own greedy_oracle / greedy_chain")
    try:
        import core.submodular_greedy as sg
        import inspect
        src = inspect.getsource(sg.greedy_oracle)
        flipped_now = bool(re.search(r"1\s*-\s*multiclass_risk", src))
        print(f"    imported core.submodular_greedy -- source currently uses "
              f"{'1 - multiclass_risk  (FLIPPED)' if flipped_now else 'multiclass_risk  (raw)'}")

        real_risk = sg.multiclass_risk

        def call_both(fn, *a, **kw):
            """Run fn as-is, then again with the sign convention inverted.
            Rebinding the module's multiclass_risk to `1 - risk` turns the
            in-source `1 - multiclass_risk(...)` back into plain risk(...),
            so this A/Bs both conventions WITHOUT editing any file."""
            as_is = fn(*a, **kw)
            sg.multiclass_risk = lambda d: 1.0 - float(real_risk(d))
            try:
                inverted = fn(*a, **kw)
            finally:
                sg.multiclass_risk = real_risk
            return as_is, inverted

        m_now, m_inv = call_both(sg.greedy_oracle, diff_sq, costs, args.lam,
                                 args.budget, free_indices)
        lbl_now = "flipped" if flipped_now else "raw"
        lbl_inv = "raw" if flipped_now else "flipped"
        print(f"    greedy_oracle  as-shipped ({lbl_now:>7}): mask={np.asarray(m_now).astype(int)}"
              f"  views={int(np.sum(m_now))}  cost={costs[np.asarray(m_now)].sum():.4f}")
        print(f"    greedy_oracle  inverted   ({lbl_inv:>7}): mask={np.asarray(m_inv).astype(int)}"
              f"  views={int(np.sum(m_inv))}  cost={costs[np.asarray(m_inv)].sum():.4f}")

        c_now, c_inv = call_both(sg.greedy_chain, means, costs, free_indices)
        print(f"    greedy_chain   as-shipped ({lbl_now:>7}): {list(c_now)}")
        print(f"    greedy_chain   inverted   ({lbl_inv:>7}): {list(c_inv)}")
        print("    -> the chains should be near mirror images: the ordering is the")
        print("       action space, so a reversed one hands the LP the wrong rungs.")
    except Exception as e:
        print(f"    (skipped: could not import core.submodular_greedy -- {type(e).__name__}: {e})")
        print("     Run from the repo root, or read sections A-C, which use a local")
        print("     copy of the same logic and need only numpy.")
        print(f"\n    local greedy_chain, add order, flipped: "
              f"{greedy_chain_local(diff_sq, costs, free_indices, f_flipped)}")
        print(f"    local greedy_chain, add order, raw:     "
              f"{greedy_chain_local(diff_sq, costs, free_indices, f_raw)}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
