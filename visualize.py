"""
visualize.py — DIO-KFC v2: ポートフォリオグレード可視化

優先度A（NOTE記事・GitHub双方で使用）:
  1. profit_curve_with_penalty.png — 割引率 vs 期待利益カーブ（ペナルティあり/なし）
  2. qini_curve.png               — evaluate_roi.py で生成済み
  3. cate_distribution.png        — evaluate_roi.py で生成済み
  4. shap_waterfall_*.png         — train_uplift.py で生成済み

優先度B（ポートフォリオの厚みを出す）:
  5. roi_comparison.png           — evaluate_roi.py で生成済み
  6. waste_reduction.png          — evaluate_roi.py で生成済み
  7. shap_summary_comparison.png  — Model A / Model B の SHAP 並列表示
  8. penalty_sensitivity.png      — λ 感度分析
  9. calibration_curves.png       — train_uplift.py で生成済み
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import japanize_matplotlib
import os

from config import (
    OUTPUT_DIR, FIGURE_DPI, KFC_RED, KFC_CREAM,
    COST_RATE, PENALTY_LAMBDA, PENALTY_LAMBDA_GRID,
    DISCOUNT_MIN, DISCOUNT_MAX, DISCOUNT_STEP,
    IRON_PLATE_THRESHOLD, SEGMENT_COLORS, RANK_COLORS, RANKS,
)

DISCOUNT_GRID = np.arange(DISCOUNT_MIN, DISCOUNT_MAX + DISCOUNT_STEP, DISCOUNT_STEP).round(2)
FEATURES_BASE = [
    "lifetime_total_mileage", "current_rank_encoded",
    "current_90d_mileage", "mileage_defense_ratio",
    "visit_momentum", "avg_spend_recent",
    "personal_cycle_days", "days_until_next_cycle",
    "past_discount_exposure_count", "wishlist_items_count",
]
FEATURES_TREATMENT = FEATURES_BASE + ["discount_rate_offered"]


def plot_profit_curve_with_penalty(
    model_a: lgb.Booster,
    model_b: lgb.Booster,
    df: pd.DataFrame,
):
    """
    割引率 vs 期待利益カーブ（ペナルティあり/なし）を3顧客パターンで描画する。
    優先度A — T-Learnerを使った場合のビジネス効果の核心を可視化。
    """
    # 3パターンの代表顧客を選定
    target = df[df["reacted_to_non_financial_incentive"] == 0].dropna(subset=["cate_at_optimal"])
    if len(target) == 0:
        return

    patterns = {
        "A: 鉄板層 (割引不要)":     target[target["p_natural_churn"] < IRON_PLATE_THRESHOLD].head(1),
        "B: 中程度説得層":           target[
            (target["cate_at_optimal"] > 0.05) & (target["cate_at_optimal"] <= 0.20)
        ].head(1),
        "C: 高弾力説得層":           target[target["cate_at_optimal"] > 0.20].head(1),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("割引率 vs 期待利益カーブ（CATEベース・ペナルティあり/なし）",
                 fontsize=14, fontweight="bold", y=1.02)

    colors = [KFC_RED, "#457B9D", "#2A9D8F"]

    for ax, (label, sample_df), color in zip(axes, patterns.items(), colors):
        if len(sample_df) == 0:
            ax.set_title(label)
            ax.text(0.5, 0.5, "該当なし", transform=ax.transAxes, ha="center")
            continue

        row     = sample_df.iloc[0]
        p_a     = model_a.predict(sample_df[FEATURES_BASE])[0]
        v       = row["avg_spend_recent"]
        exposure = row["past_discount_exposure_count"]

        profits_no_penalty = []
        profits_with_penalty = []

        for d in DISCOUNT_GRID:
            X_tmp = sample_df[FEATURES_TREATMENT].copy()
            X_tmp["discount_rate_offered"] = d
            p_b = model_b.predict(X_tmp)[0]
            cate = p_a - p_b

            # ペナルティなし
            p_no  = cate * v * (1 - COST_RATE) - d * v * (1 - p_b)
            # ペナルティあり
            penalty = PENALTY_LAMBDA * v * (d ** 2) * exposure
            p_with = p_no - penalty

            profits_no_penalty.append(p_no)
            profits_with_penalty.append(p_with)

        profits_no_penalty  = np.array(profits_no_penalty)
        profits_with_penalty = np.array(profits_with_penalty)

        opt_d_no   = DISCOUNT_GRID[np.argmax(profits_no_penalty)]
        opt_d_with = DISCOUNT_GRID[np.argmax(profits_with_penalty)]

        ax.plot(DISCOUNT_GRID * 100, profits_no_penalty,
                color=color, linewidth=2, linestyle="--", alpha=0.6,
                label=f"ペナルティなし (d*={opt_d_no:.0%})")
        ax.plot(DISCOUNT_GRID * 100, profits_with_penalty,
                color=color, linewidth=2.5, label=f"ペナルティあり (d*={opt_d_with:.0%})")

        ax.axvline(opt_d_with * 100, color=color, linewidth=1, linestyle=":",
                   alpha=0.8)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")

        ax.set_xlabel("割引率 (%)")
        ax.set_ylabel("期待利益（円）")
        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # 顧客情報注釈
        ax.text(0.98, 0.03,
                f"P_A={p_a:.2f}\nV=¥{v:,}\nExposure={int(exposure)}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "profit_curve_with_penalty.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


def plot_penalty_sensitivity(sensitivity_df: pd.DataFrame):
    """
    ペナルティ強度 λ の感度分析を2軸折れ線グラフで描画する（優先度B）。
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_profit   = KFC_RED
    color_discount = "#457B9D"

    ax1.plot(sensitivity_df["lambda"], sensitivity_df["avg_discount"] * 100,
             color=color_discount, marker="o", linewidth=2, label="平均最適割引率 (%)")
    ax1.set_xlabel("ペナルティ強度 λ", fontsize=11)
    ax1.set_ylabel("平均最適割引率 (%)", color=color_discount, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_discount)

    ax2 = ax1.twinx()
    ax2.plot(sensitivity_df["lambda"], sensitivity_df["total_profit"],
             color=color_profit, marker="s", linewidth=2, linestyle="--",
             label="総期待利益 (円)")
    ax2.set_ylabel("総期待利益 (円)", color=color_profit, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_profit)

    # ゼロ割引率の線
    ax1.plot(sensitivity_df["lambda"], sensitivity_df["pct_zero_discount"] * 100,
             color="gray", marker="^", linewidth=1.5, linestyle=":", alpha=0.8,
             label="ゼロ割引率顧客 (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    ax1.set_title("ペナルティ強度 λ の感度分析\n（平均最適割引率 & 総期待利益への影響）",
                  fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "penalty_sensitivity.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


def plot_discount_distribution(df_result: pd.DataFrame):
    """最適割引率の分布（ランク別）を描画する（優先度B）。"""
    target = df_result[
        (df_result["reacted_to_non_financial_incentive"] == 0) &
        (df_result["optimal_discount"].notna())
    ].copy()

    fig, axes = plt.subplots(1, len(RANKS), figsize=(16, 4), sharey=True)
    fig.suptitle("最適割引率の分布 — ランク別（AI最適化）",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, rank in zip(axes, RANKS):
        sub = target[target["current_rank"] == rank]["optimal_discount"] * 100
        color = RANK_COLORS[rank]
        ax.hist(sub, bins=20, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(sub.mean(), color="black", linewidth=1.5, linestyle="--",
                   label=f"平均={sub.mean():.1f}%")
        ax.set_title(rank, fontweight="bold", fontsize=10)
        ax.set_xlabel("割引率 (%)")
        if ax == axes[0]:
            ax.set_ylabel("顧客数")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "discount_distribution.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  保存: {path}")


def run_visualization_pipeline(
    model_a: lgb.Booster,
    model_b: lgb.Booster,
    df: pd.DataFrame,
    df_result: pd.DataFrame,
    sensitivity_df: pd.DataFrame | None = None,
):
    """全図表を一括生成する。"""
    print("\n" + "=" * 60)
    print("🎨 可視化パイプライン")
    print("=" * 60)

    print("\n  [優先度A]")
    plot_profit_curve_with_penalty(model_a, model_b, df_result)

    print("\n  [優先度B]")
    if sensitivity_df is not None:
        plot_penalty_sensitivity(sensitivity_df)
    plot_discount_distribution(df_result)

    print("\n✅ 可視化 完了")


if __name__ == "__main__":
    from generate_data import generate_data, save_data
    from train_uplift import run_training_pipeline
    from optimize_incentive import run_optimization_pipeline, sensitivity_analysis
    from evaluate_roi import run_evaluation_pipeline

    df = generate_data()
    save_data(df)
    model_a, model_b, df_cate, metrics_train, qini = run_training_pipeline(df)
    df_result = run_optimization_pipeline(model_a, model_b, df_cate)
    metrics = run_evaluation_pipeline(df_result, qini)
    sensitivity_df = sensitivity_analysis(model_a, model_b, df_cate)
    run_visualization_pipeline(model_a, model_b, df_cate, df_result, sensitivity_df)
