"""
evaluate_roi.py — DIO-KFC v2: ビジネス評価・増分ROI

評価指標:
  - 無駄なインセンティブ削減率（鉄板層への割引カット効果）
  - 増分利益 Incremental ROI（一律20% vs AI最適化）
  - 非金銭防衛成功率（ベンチマーク: 30%）
  - CATE分布（顧客タイプ別の分離度）
  - Qini曲線の描画
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib
import os

from config import (
    OUTPUT_DIR, UNIFORM_DISCOUNT_RATE, COST_RATE,
    IRON_PLATE_THRESHOLD, NON_FINANCIAL_REACTION_RATE,
    FIGURE_DPI, KFC_RED, CATE_COLORS,
    RANK_COLORS, RANKS,
)


def compute_business_metrics(df_result: pd.DataFrame) -> dict:
    """
    ビジネス評価指標を計算して返す。

    Parameters
    ----------
    df_result: optimize_incentive.run_optimization_pipeline() の出力df

    Returns
    -------
    dict: 各評価指標の値
    """
    total = len(df_result)
    target = df_result[df_result["reacted_to_non_financial_incentive"] == 0].copy()
    n_target = len(target)

    # ── 1. 非金銭防衛成功率 ─────────────────────────────────
    n_non_fin_success = (df_result["reacted_to_non_financial_incentive"] == 1).sum()
    non_financial_success_rate = n_non_fin_success / total

    # ── 2. 無駄なインセンティブ削減率 ──────────────────────
    iron_plate = target[target["is_iron_plate"].fillna(False) == True]
    n_iron = len(iron_plate)

    # 一律割引時に鉄板層に出していたはずのコスト
    uniform_cost_iron = (
        iron_plate["avg_spend_recent"] * UNIFORM_DISCOUNT_RATE *
        (1 - iron_plate["p_natural_churn"].fillna(0.5))
    ).sum()

    # AI最適化での鉄板層割引コスト（d*=0 なのでゼロ）
    ai_cost_iron = 0.0

    total_uniform_cost = (
        target["avg_spend_recent"] * UNIFORM_DISCOUNT_RATE *
        (1 - target["p_natural_churn"].fillna(0.5))
    ).sum()

    waste_reduction_rate = uniform_cost_iron / total_uniform_cost if total_uniform_cost > 0 else 0

    # ── 3. 増分利益（Incremental ROI） ──────────────────────
    total_uniform_profit = target["uniform_profit"].sum()
    total_ai_profit      = target["optimal_profit"].sum()
    incremental_roi      = total_ai_profit - total_uniform_profit

    # ── 4. 顧客タイプ別統計 ──────────────────────────────────
    type_stats = target.groupby("customer_type").agg(
        顧客数=("user_id", "count"),
        平均CATE=("cate_at_optimal", "mean"),
        平均最適割引率=("optimal_discount", "mean"),
        平均期待利益=("optimal_profit", "mean"),
    ).round(3)

    # ── 5. ランク別削減効果 ──────────────────────────────────
    rank_stats = target.groupby("current_rank").agg(
        顧客数=("user_id", "count"),
        鉄板層比率=("is_iron_plate", "mean"),
        平均最適割引率=("optimal_discount", "mean"),
        平均CATE=("cate_at_optimal", "mean"),
        AI利益合計=("optimal_profit", "sum"),
        一律利益合計=("uniform_profit", "sum"),
    ).round(3)
    rank_stats["利益増分"] = (rank_stats["AI利益合計"] - rank_stats["一律利益合計"]).round(0)

    metrics = {
        "total_customers":             total,
        "target_customers":            n_target,
        "non_financial_success_rate":  non_financial_success_rate,
        "n_iron_plate":                n_iron,
        "iron_plate_rate":             n_iron / n_target,
        "waste_reduction_rate":        waste_reduction_rate,
        "uniform_cost_iron":           uniform_cost_iron,
        "total_uniform_profit":        total_uniform_profit,
        "total_ai_profit":             total_ai_profit,
        "incremental_roi":             incremental_roi,
        "type_stats":                  type_stats,
        "rank_stats":                  rank_stats,
    }

    # ── サマリー出力 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📈 ビジネス評価指標")
    print("=" * 60)
    print(f"\n  ① 非金銭防衛成功率:       {non_financial_success_rate:.1%}"
          f"  (ベンチマーク: {NON_FINANCIAL_REACTION_RATE:.0%})")
    print(f"  ② 鉄板層比率:             {n_iron/n_target:.1%}"
          f" ({n_iron:,}人)")
    print(f"  ③ 無駄なインセンティブ削減率: {waste_reduction_rate:.1%}")
    print(f"  ④ 一律{UNIFORM_DISCOUNT_RATE:.0%}の総期待利益:  {total_uniform_profit:,.0f}円")
    print(f"  ⑤ AI最適化の総期待利益:   {total_ai_profit:,.0f}円")
    print(f"  ⑥ 増分利益 (Incr. ROI):  +{incremental_roi:,.0f}円"
          f" ({incremental_roi/abs(total_uniform_profit):.1%}改善)")

    print("\n  【顧客タイプ別統計】")
    print(type_stats.to_string())

    print("\n  【ランク別統計】")
    print(rank_stats.to_string())

    return metrics


def run_evaluation_pipeline(
    df_result: pd.DataFrame,
    qini: dict | None = None,
) -> dict:
    """評価指標の計算 + 全ビジネス可視化を実行する。"""
    metrics = compute_business_metrics(df_result)

    # Qini曲線の描画
    if qini is not None:
        _plot_qini_curve(qini)

    # CATE分布
    _plot_cate_distribution(df_result)

    # ROI比較
    _plot_roi_comparison(metrics)

    # 無駄な割引カット率（ランク別）
    _plot_waste_reduction_by_rank(metrics["rank_stats"])

    print("\n✅ 評価・可視化 完了\n")
    return metrics


# ─── 可視化関数 ───────────────────────────────────────────────

def _plot_qini_curve(qini: dict):
    """Qini曲線を描画する（優先度A）。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        qini["proportions"], qini["uplifts"],
        color=KFC_RED, linewidth=2.5, label=f"T-Learner (Qini係数={qini['qini_coef']:.4f})"
    )
    ax.plot(
        qini["proportions"], qini["random_line"],
        linestyle="--", color="gray", linewidth=1.5, label="ランダム施策"
    )
    ax.fill_between(
        qini["proportions"],
        qini["uplifts"], qini["random_line"],
        alpha=0.15, color=KFC_RED, label="Upliftモデルの優位性"
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("介入対象の割合 (%)", fontsize=11)
    ax.set_ylabel("累積 Uplift（離脱率低下量）", fontsize=11)
    ax.set_title("Qini曲線 — T-Learner vs ランダム施策", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "qini_curve.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


def _plot_cate_distribution(df_result: pd.DataFrame):
    """CATE分布ヒストグラムを顧客タイプ別に色分けして描画する（優先度A）。"""
    target = df_result[df_result["reacted_to_non_financial_incentive"] == 0].copy()
    target = target.dropna(subset=["cate_at_optimal"])

    fig, ax = plt.subplots(figsize=(9, 5))

    for ctype, color in CATE_COLORS.items():
        sub = target[target["customer_type"] == ctype]["cate_at_optimal"]
        if len(sub) == 0:
            continue
        ax.hist(
            sub, bins=40, alpha=0.65, color=color, edgecolor="white",
            linewidth=0.4, label=f"{ctype} (n={len(sub):,})"
        )

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="CATE = 0")
    ax.axvline(0.05, color="gray", linewidth=1, linestyle=":",
               label=f"説得層閾値 (CATE ≥ 0.05)")
    ax.set_xlabel("CATE（割引による離脱確率低下量）", fontsize=11)
    ax.set_ylabel("顧客数", fontsize=11)
    ax.set_title("CATE分布 — 顧客タイプ別（鉄板層・説得層・天邪鬼層）",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cate_distribution.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


def _plot_roi_comparison(metrics: dict):
    """ROI比較棒グラフを描画する（優先度B）。"""
    labels  = [f"一律{UNIFORM_DISCOUNT_RATE:.0%}割引", "AI最適化 (CATE)"]
    values  = [metrics["total_uniform_profit"], metrics["total_ai_profit"]]
    colors  = ["#A0A0A0", KFC_RED]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(max(values)) * 0.01,
                f"¥{val:,.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    incr = metrics["incremental_roi"]
    ax.annotate(
        f"+¥{incr:,.0f}\n({incr/abs(values[0]):.1%}改善)",
        xy=(1, values[1]), xytext=(1.3, (values[0] + values[1]) / 2),
        arrowprops=dict(arrowstyle="->", color=KFC_RED),
        fontsize=10, color=KFC_RED, fontweight="bold",
    )
    ax.set_ylabel("総期待利益（円）", fontsize=11)
    ax.set_title("増分利益 (Incremental ROI) — 一律割引 vs AI最適化",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roi_comparison.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


def _plot_waste_reduction_by_rank(rank_stats: pd.DataFrame):
    """ランク別の無駄な割引カット率を積み上げ棒で描画する（優先度B）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: 鉄板層比率（ランク別）
    ranks_ordered = [r for r in RANKS if r in rank_stats.index]
    iron_rates    = [rank_stats.loc[r, "鉄板層比率"] if r in rank_stats.index else 0
                     for r in ranks_ordered]
    colors = [RANK_COLORS[r] for r in ranks_ordered]

    axes[0].bar(ranks_ordered, iron_rates, color=colors, edgecolor="white")
    for i, (r, v) in enumerate(zip(ranks_ordered, iron_rates)):
        axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("鉄板層比率 (割引ゼロ顧客の割合)")
    axes[0].set_title("ランク別 鉄板層比率", fontweight="bold")
    axes[0].set_ylim(0, min(max(iron_rates) * 1.3, 1.0))
    axes[0].grid(axis="y", alpha=0.3)

    # Right: 利益増分（ランク別）
    profit_diffs = [rank_stats.loc[r, "利益増分"] if r in rank_stats.index else 0
                    for r in ranks_ordered]
    bar_colors = [KFC_RED if v >= 0 else "#A0A0A0" for v in profit_diffs]
    axes[1].bar(ranks_ordered, profit_diffs, color=bar_colors, edgecolor="white")
    for i, v in enumerate(profit_diffs):
        axes[1].text(i, v + max(profit_diffs) * 0.01 if v >= 0 else v - max(profit_diffs) * 0.05,
                     f"¥{v:,.0f}", ha="center", va="bottom", fontsize=8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("利益増分（円）")
    axes[1].set_title("ランク別 利益増分（AI − 一律割引）", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("無駄な割引カット効果 — ランク別分析", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "waste_reduction.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


if __name__ == "__main__":
    from generate_data import generate_data, save_data
    from train_uplift import run_training_pipeline
    from optimize_incentive import run_optimization_pipeline

    df = generate_data()
    save_data(df)
    model_a, model_b, df_cate, metrics_train, qini = run_training_pipeline(df)
    df_result = run_optimization_pipeline(model_a, model_b, df_cate)
    metrics = run_evaluation_pipeline(df_result, qini)
