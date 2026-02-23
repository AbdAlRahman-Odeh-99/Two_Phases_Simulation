"""
Visualization utilities for experiment results.
"""

import matplotlib.pyplot as plt
from typing import List, Dict


def plot_phase1_loss_vs_tph1(results: List[Dict]) -> None:
    """
    Plot Phase 1 prediction loss vs. offline initialization length.
    
    Args:
        results: List of result dictionaries from experiments
    """
    T_vals = [r["T_ph1"] for r in results]
    ph1_losses = [r["ph1_prediction_loss"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(T_vals, ph1_losses, marker='o', linewidth=2, markersize=6)
    plt.xlabel(r"$T_{\mathrm{ph1}}$", fontsize=12)
    plt.ylabel("Phase-1 Prediction Loss", fontsize=12)
    plt.title("Phase-1 Loss vs Offline Initialization Length", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_phase2_loss_vs_tph1(results: List[Dict]) -> None:
    """
    Plot Phase 2 prediction loss vs. offline initialization length.
    
    Args:
        results: List of result dictionaries from experiments
    """
    T_vals = [r["T_ph1"] for r in results]
    ph2_losses = [r["ph2_prediction_loss"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(T_vals, ph2_losses, marker='o', linewidth=2, markersize=6)
    plt.xlabel(r"$T_{\mathrm{ph1}}$", fontsize=12)
    plt.ylabel("Phase-2 Prediction Loss", fontsize=12)
    plt.title("Phase-2 Loss vs Offline Initialization Length", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_total_loss_vs_tph1(
    results: List[Dict],
    lambda_cost: float,
    save_path: str = None
) -> None:
    """
    Plot total prediction loss vs. offline initialization length.
    
    Args:
        results: List of result dictionaries from experiments
        lambda_cost: Cost parameter (used in plot title)
        save_path: Optional path to save the figure
    """
    T_vals = [r["T_ph1"] for r in results]
    total_losses = [r["total_prediction_loss"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(T_vals, total_losses, marker='o', linewidth=2, markersize=6)
    plt.xlabel(r"$T_{\mathrm{ph1}}$", fontsize=12)
    plt.ylabel("Total Prediction Loss", fontsize=12)
    plt.title("Total Prediction Loss vs Offline Initialization Length", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.1, 0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_accuracy_comparison(results: List[Dict]) -> None:
    """
    Plot Phase 1 and Phase 2 accuracies comparison.
    
    Args:
        results: List of result dictionaries from experiments
    """
    T_vals = [r["T_ph1"] for r in results]
    acc_ph1 = [r["acc_ph1"] for r in results]
    acc_ph2 = [r["acc_ph2"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(T_vals, acc_ph1, marker='o', label='Phase 1', linewidth=2, markersize=6)
    plt.plot(T_vals, acc_ph2, marker='s', label='Phase 2', linewidth=2, markersize=6)
    plt.xlabel(r"$T_{\mathrm{ph1}}$", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Phase Accuracy Comparison", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
