# -*- coding: utf-8 -*-
"""
core/arm_elimination.py
"""

from __future__ import annotations
import numba
import numpy as np
import scipy.optimize as opt

def arm_elimination_checkpoints(T, min_remaining=10):
    """
    Cumulative checkpoints for epoch lengths T/2, T/4, T/8, ...

    A checkpoint is omitted when fewer than `min_remaining` rounds
    would remain afterward. The final round is never a checkpoint.
    """
    T = int(T)
    if T <= 1:
        return []

    checkpoints = []
    completed = 0
    epoch = T // 2

    while epoch >= 1:
        next_completed = completed + epoch
        remaining = T - next_completed

        if next_completed >= T or remaining < min_remaining:
            break

        checkpoints.append(next_completed)
        completed = next_completed
        epoch //= 2

    return checkpoints

def restrict_candidates(candidate_idx, active_arms):
    candidate_idx = np.asarray(candidate_idx, dtype=int)
    if active_arms is not None:
        candidate_idx = candidate_idx[active_arms[candidate_idx]]
    if candidate_idx.size == 0:
        raise RuntimeError("No active candidate arms remain.")
    return candidate_idx

def solve_elimination_lp(rewards, costs, budget_per_round):
    rewards = np.asarray(rewards, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    n = len(rewards)
    res = opt.linprog(
        -rewards,
        A_ub=costs.reshape(1, -1),
        b_ub=np.array([max(0.0, float(budget_per_round))]),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Arm-elimination LP failed: {res.message}")
    p = np.clip(res.x, 0.0, None)
    if p.sum() > 0:
        p /= p.sum()
    return p

def eliminate_arms_ucb_lcb(r_hat, combo_counts, combo_cost, active_arms, alpha_ucb, round_idx, budget_per_round, tol=1e-9,):
    """
    Eliminate arms using UCB- and LCB-based LP solutions.
    Only currently active arms participate.
    Returns
    -------
    new_active : bool ndarray
    info : dict
    """
    active_arms = np.asarray(active_arms, dtype=bool)
    active_idx = np.flatnonzero(active_arms)

    if active_idx.size == 0:
        raise ValueError("Arm elimination received an empty active-arm set.")
    if len(active_idx) <= 1:
        return active_arms.copy(), {"before": len(active_idx), "after": len(active_idx), "eliminated": 0,}
    if np.any(combo_counts[active_idx] <= 0):
        raise ValueError("All active arm counts must be positive.")
    

    bonus = np.sqrt(alpha_ucb * np.log(round_idx + 2) / combo_counts[active_idx])
    ucb = r_hat[active_idx] + bonus
    lcb = r_hat[active_idx] - bonus
    costs = combo_cost[active_idx]

    p_ucb = solve_elimination_lp(ucb, costs, budget_per_round,)
    p_lcb = solve_elimination_lp(lcb, costs, budget_per_round,)
    survive_local = (p_ucb > tol) | (p_lcb > tol)

    new_active = np.zeros_like(active_arms)
    new_active[active_idx[survive_local]] = True

    # The cheapest/free arm must remain available so future LPs stay feasible.
    min_cost = float(np.min(combo_cost))
    protected = np.isclose(combo_cost, min_cost, atol=tol, rtol=0.0)
    new_active[protected] = True

    # Never allow the entire action space to disappear.
    if not new_active.any():
        new_active[active_idx[np.argmax(ucb)]] = True

    return new_active, {"before": int(len(active_idx)), "after": int(new_active.sum()), "eliminated": int(len(active_idx) - new_active.sum()),}