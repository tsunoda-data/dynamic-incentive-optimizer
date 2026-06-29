"""
optimize_incentive.py — DIO-KFC v2: CATEベース期待利益最大化

T-Learnerで算出したCATEを利益計算の中核に据え、
最適割引率 d* をグリッドサーチで算出する。

目的関数:
  Profit(d) = CATE(d) × V × (1-C)
              - d × V × (1 - P_B(churn|d))
              - Penalty(d)

  CATE(d) = P_A(churn) - P_B(churn|d)
  Penalty(d) = λ × V × d² × past_discount_exposure_count

なぜT-Learnerを使うのか:
  - CATEがゼロの鉄板層では第1項もゼロになり d*=0 が自動選択される
  - 「割引の純粋な増分効果」だけを最適化の軸に据えることで、
    T-Learner構築の必然性が利益計算本体に直結する
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    RANDOM_SEED, OUTPUT_DIR,
    COST_RATE, PENALTY_LAMBDA, PENALTY_LAMBDA_GRID,
    DISCOUNT_MIN, DISCOUNT_MAX, DISCOUNT_STEP,
    IRON_PLATE_THRESHOLD, UNIFORM_DISCOUNT_RATE,
)

# 特徴量リストを直接定義（循環インポート回避）
FEATURES_BASE = [
    "lifetime_total_mileage", "current_rank_encoded",
    "current_90d_mileage", "mileage_defense_ratio",
    "visit_momentum", "avg_spend_recent",
    "personal_cycle_days", "days_until_next_cycle",
    "past_discount_exposure_count", "wishlist_items_count",
]
DISCOUNT_FEATURE = "discount_rate_offered"
FEATURES_TREATMENT = FEATURES_BASE + [DISCOUNT_FEATURE]

DISCOUNT_GRID = np.arange(DISCOUNT_MIN, DISCOUNT_MAX + DISCOUNT_STEP, DISCOUNT_STEP).round(2)


def compute_expected_profit(
    p_a: float | np.ndarray,
    p_b_grid: np.ndarray,
    v: float | np.ndarray,
    d_grid: np.ndarray,
    exposure: float | np.ndarray,
    lambda_: float = PENALTY_LAMBDA,
    cost_rate: float = COST_RATE,
) -> np.ndarray:
    """
    割引率グリッド全体の期待利益を計算する（ベクトル化）。

    Parameters
    ----------
    p_a     : 自然離脱確率 P_A(churn) — shape (n,)
    p_b_grid: 各割引率での P_B(churn|d) — shape (n, n_discount)
    v       : 客単価 — shape (n,)
    d_grid  : 割引率グリッド — shape (n_discount,)
    exposure: クーポン接触回数 — shape (n,)

    Returns
    -------
    profit : shape (n, n_discount)
    """
    n = len(p_a)
    # Broadcasting: (n,) → (n, 1) で (n, n_discount) にブロードキャスト
    p_a_2d       = p_a[:, np.newaxis]          # (n, 1)
    p_b_2d       = p_b_grid                    # (n, n_discount)
    v_2d         = v[:, np.newaxis]             # (n, 1)
    exposure_2d  = exposure[:, np.newaxis]      # (n, 1)
    d_2d         = d_grid[np.newaxis, :]        # (1, n_discount)

    # CATE(d) = P_A - P_B(d)
    cate = p_a_2d - p_b_2d                     # (n, n_discount)

    # 第1項: CATE × V × (1 - C)    → 増分粗利
    incremental_margin = cate * v_2d * (1 - cost_rate)

    # 第2項: d × V × (1 - P_B(d)) → 割引コスト（来店者への割引支払い）
    discount_cost = d_2d * v_2d * (1 - p_b_2d)

    # 第3項: λ × V × d² × E       → 割引慣れペナルティ
    penalty = lambda_ * v_2d * (d_2d ** 2) * exposure_2d

    profit = incremental_margin - discount_cost - penalty
    return profit


def _predict_p_b_grid(model_b: lgb.Booster, X_base: pd.DataFrame) -> np.ndarray:
    """
    全割引率グリッドに対してModel Bの予測確率を計算する。

    Returns
    -------
    p_b_grid: shape (n, n_discount)
    """
    n = len(X_base)
    n_d = len(DISCOUNT_GRID)
    p_b_grid = np.zeros((n, n_d))

    for j, d in enumerate(DISCOUNT_GRID):
        X_tmp = X_base.copy()
        X_tmp[DISCOUNT_FEATURE] = d
        p_b_grid[:, j] = model_b.predict(X_tmp[FEATURES_TREATMENT])

    return p_b_grid


def run_optimization_pipeline(
    model_a: lgb.Booster,
    model_b: lgb.Booster,
    df: pd.DataFrame,
    lambda_: float = PENALTY_LAMBDA,
) -> pd.DataFrame:
    """
    全非反応者に対して最適割引率 d* を算出する。

    Returns
    -------
    df_result: 元のdfに最適化結果を付与したDataFrame
    """
    print("\n" + "=" * 60)
    print("⚙️  Stage 2: CATEベース期待利益最大化")
    print("=" * 60)
    print(f"  ペナルティ強度 λ = {lambda_}")
    print(f"  割引率グリッド: {DISCOUNT_GRID[0]:.2f} 〜 {DISCOUNT_GRID[-1]:.2f}"
          f" ({len(DISCOUNT_GRID)}点)")

    # ── 最適化対象: 非反応者のみ ─────────────────────────────
    target_mask = df["reacted_to_non_financial_incentive"] == 0
    df_target   = df[target_mask].copy().reset_index(drop=True)
    print(f"  最適化対象: {len(df_target):,}人")

    X_base = df_target[FEATURES_BASE]
    v      = df_target["avg_spend_recent"].values.astype(float)
    exposure = df_target["past_discount_exposure_count"].values.astype(float)

    # Model A: 自然離脱確率
    p_a = model_a.predict(X_base)

    # Model B: 全割引率グリッドの予測確率
    print("  Model B グリッド予測中...")
    p_b_grid = _predict_p_b_grid(model_b, X_base)

    # ── 期待利益の計算 ───────────────────────────────────────
    profit_matrix = compute_expected_profit(
        p_a, p_b_grid, v, DISCOUNT_GRID, exposure, lambda_
    )  # shape: (n, n_discount)

    # 最適割引率のインデックス
    opt_idx = np.argmax(profit_matrix, axis=1)

    optimal_discount = DISCOUNT_GRID[opt_idx]
    optimal_profit   = profit_matrix[np.arange(len(df_target)), opt_idx]

    # ── 鉄板層の自動カット ──────────────────────────────────
    # P_A < 0.10 の顧客は割引不要 → d* = 0 を強制
    iron_plate_mask = p_a < IRON_PLATE_THRESHOLD
    optimal_discount[iron_plate_mask] = 0.0

    # d=0 の利益を鉄板層の実際の利益として再計算
    p_b_at_zero = p_b_grid[:, 0]  # d=0 列
    cate_at_zero = p_a - p_b_at_zero
    profit_at_zero = cate_at_zero * v * (1 - COST_RATE)
    optimal_profit[iron_plate_mask] = profit_at_zero[iron_plate_mask]

    n_iron = iron_plate_mask.sum()
    print(f"  鉄板層 (P_A < {IRON_PLATE_THRESHOLD}): {n_iron:,}人"
          f" ({n_iron/len(df_target):.1%}) → 割引ゼロに強制")

    # ── 一律割引との比較 ─────────────────────────────────────
    uniform_idx = np.argmin(np.abs(DISCOUNT_GRID - UNIFORM_DISCOUNT_RATE))
    uniform_profit = profit_matrix[:, uniform_idx]
    profit_improvement = optimal_profit - uniform_profit

    # ── CATE値の格納 ─────────────────────────────────────────
    cate_at_optimal = np.array([
        p_a[i] - p_b_grid[i, opt_idx[i]] for i in range(len(df_target))
    ])

    # ── 顧客タイプの分類 ──────────────────────────────────────
    def classify_cate(cate_val):
        if cate_val < -0.02:
            return "天邪鬼層 (Sleeping Dogs)"
        elif cate_val < 0.05:
            return "鉄板層 (Sure Things)"
        else:
            return "説得層 (Persuadables)"

    customer_type = [classify_cate(c) for c in cate_at_optimal]

    df_target = df_target.assign(
        p_natural_churn   = p_a,
        optimal_discount  = optimal_discount.round(2),
        optimal_profit    = optimal_profit.round(1),
        uniform_profit    = uniform_profit.round(1),
        profit_improvement = profit_improvement.round(1),
        cate_at_optimal   = cate_at_optimal.round(4),
        is_iron_plate     = iron_plate_mask,
        customer_type     = customer_type,
    )

    # ── サマリー出力 ─────────────────────────────────────────
    print(f"\n【最適化結果サマリー】")
    print(f"  平均最適割引率:      {optimal_discount.mean():.1%}")
    print(f"  一律{UNIFORM_DISCOUNT_RATE:.0%}との利益差:   "
          f"{profit_improvement.sum():,.0f}円 (合計)")
    print(f"  割引ゼロ(d*=0):     "
          f"{(optimal_discount == 0).sum():,}人 "
          f"({(optimal_discount == 0).mean():.1%})")

    type_dist = pd.Series(customer_type).value_counts()
    print("\n  顧客タイプ分布:")
    for t, n in type_dist.items():
        print(f"    {t}: {n:,}人 ({n/len(df_target):.1%})")

    # 元のdfに結果をマージ
    df_result = df.copy()
    cols_to_add = [
        "p_natural_churn", "optimal_discount", "optimal_profit",
        "uniform_profit", "profit_improvement", "cate_at_optimal",
        "is_iron_plate", "customer_type",
    ]
    for col in cols_to_add:
        df_result[col] = np.nan if col not in ["customer_type", "is_iron_plate"] else None

    df_result.loc[target_mask, cols_to_add] = df_target[cols_to_add].values

    print("\n✅ Stage 2 完了\n")
    return df_result


def sensitivity_analysis(
    model_a: lgb.Booster,
    model_b: lgb.Booster,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    ペナルティ強度 λ の感度分析を実行する。

    Returns
    -------
    pd.DataFrame: λ別の平均割引率・合計期待利益
    """
    print("\n📊 感度分析: ペナルティ強度 λ の影響")
    rows = []
    for lam in PENALTY_LAMBDA_GRID:
        df_res = run_optimization_pipeline(model_a, model_b, df, lambda_=lam)
        target = df_res[df_res["reacted_to_non_financial_incentive"] == 0]
        rows.append({
            "lambda":           lam,
            "avg_discount":     target["optimal_discount"].mean(),
            "total_profit":     target["optimal_profit"].sum(),
            "pct_zero_discount": (target["optimal_discount"] == 0).mean(),
        })
        print(f"  λ={lam:.1f}: 平均割引率={rows[-1]['avg_discount']:.1%} | "
              f"合計利益={rows[-1]['total_profit']:,.0f}円 | "
              f"ゼロ割引={rows[-1]['pct_zero_discount']:.1%}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from generate_data import generate_data, save_data
    from train_uplift import run_training_pipeline

    df = generate_data()
    save_data(df)
    model_a, model_b, df_with_cate, metrics, qini = run_training_pipeline(df)
    df_result = run_optimization_pipeline(model_a, model_b, df_with_cate)
