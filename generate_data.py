"""
generate_data.py — DIO-KFC v2: 合成データ生成

10,000人の顧客データを以下の仕様で生成する:
  - 4セグメント (ファミリー/単身ヘビー/学生/シニア)
  - 90日ローリング判定対応のランク閾値
  - 段階的介入フロー:
      サイクル崩れ検知 → 非金銭インセンティブ(30%反応)
      → 非反応者(70%)にRCT割り当て(50:50)
  - is_churn の定義: 「90日間でランク閾値マイルに届かなかった」
"""

import numpy as np
import pandas as pd
from config import (
    RANDOM_SEED, N_CUSTOMERS,
    RANKS, RANK_THRESHOLDS, RANK_ENCODING, RANK_DISTRIBUTION,
    SEGMENTS, SEGMENT_CONFIG,
    BASE_CHURN_RATE, NON_FINANCIAL_REACTION_RATE,
    RCT_TREATMENT_RATIO, IRON_PLATE_RATIO,
    DISCOUNT_MIN, DISCOUNT_MAX,
    DATA_DIR,
)
import os


def generate_data(n: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    合成顧客データを生成する。

    Returns
    -------
    pd.DataFrame
        16カラム + user_id の全17カラムを含む顧客データ
    """
    rng = np.random.default_rng(seed)

    # ── 1. セグメント割り当て ──────────────────────────────
    seg_probs = [SEGMENT_CONFIG[s]["ratio"] for s in SEGMENTS]
    segments  = rng.choice(SEGMENTS, size=n, p=seg_probs)

    # ── 2. ランク割り当て（セグメントと独立） ───────────────
    rank_probs = [RANK_DISTRIBUTION[r] for r in RANKS]
    ranks      = rng.choice(RANKS, size=n, p=rank_probs)

    # ── 3. 基本特徴量の生成 ───────────────────────────────
    lifetime_total_mileage = np.zeros(n, dtype=int)
    avg_spend_recent       = np.zeros(n, dtype=int)
    personal_cycle_days    = np.zeros(n, dtype=int)
    price_elasticity       = np.zeros(n)

    for seg in SEGMENTS:
        mask = segments == seg
        cfg  = SEGMENT_CONFIG[seg]

        # 生涯累計マイル: ランクが高いほど高め
        rank_multiplier = np.array([RANK_ENCODING[r] + 1 for r in ranks[mask]])
        lifetime_total_mileage[mask] = rng.integers(
            200 * rank_multiplier, 5000 * rank_multiplier + 1
        )

        # 直近3ヶ月の平均客単価
        lo, hi = cfg["spend_range"]
        avg_spend_recent[mask] = rng.integers(lo, hi + 1, size=mask.sum())

        # 個人来店周期（AIが算出した値を模擬）
        lo_c, hi_c = cfg["cycle_range"]
        personal_cycle_days[mask] = rng.integers(lo_c, hi_c + 1, size=mask.sum())

        # 価格弾力性（セグメント依存）
        price_elasticity[mask] = cfg["price_elasticity"]

    # ── 4. 直近90日マイル（ランク防衛状況を模擬） ────────────
    current_90d_mileage = np.zeros(n, dtype=int)
    for rank in RANKS:
        mask      = ranks == rank
        threshold = RANK_THRESHOLDS[rank]
        if threshold == 0:
            # レギュラーはランク防衛不要 → マイル0〜29の範囲
            current_90d_mileage[mask] = rng.integers(0, 30, size=mask.sum())
        else:
            # 防衛中: 50〜130%の範囲でランダム（一部は危機状態）
            ratios = rng.uniform(0.40, 1.30, size=mask.sum())
            current_90d_mileage[mask] = np.clip(
                (ratios * threshold).astype(int), 0, threshold * 2
            )

    # ── 5. 派生特徴量 ─────────────────────────────────────
    mileage_defense_ratio = np.where(
        ranks == "レギュラー",
        999.0,  # sentinel: 防衛不要
        np.clip(
            current_90d_mileage / np.array([max(RANK_THRESHOLDS[r], 1) for r in ranks]),
            0.0, 3.0
        )
    )

    # 来店モメンタム: 1.0基準、0.7未満は危機
    visit_momentum = rng.uniform(0.4, 1.8, size=n).round(2)

    # 次の個人サイクル期限までの残日数（負=超過）
    # 40%程度を「瀬戸際〜超過」とする
    is_at_risk = rng.random(n) < 0.40
    days_until_next_cycle = np.where(
        is_at_risk,
        rng.integers(-10, 5, size=n),       # 危機: -10〜+4日
        rng.integers(5, int(np.max(personal_cycle_days)) + 1, size=n)  # 余裕あり
    )

    # クーポン接触回数（過去90日）
    past_discount_exposure_count = rng.integers(0, 12, size=n)

    # ウィッシュリスト登録商品数
    wishlist_items_count = rng.integers(0, 8, size=n)

    # 配信連携用（モデル非入力）
    optimal_send_hour = rng.integers(8, 22, size=n)

    # ── 6. 段階的介入フラグの生成 ─────────────────────────
    # Step 3: 非金銭インセンティブ反応（全体の30%）
    # 反応率はウィッシュリスト登録数・モメンタムに正の相関
    non_financial_score = (
        0.4 * (wishlist_items_count / 7.0) +
        0.4 * np.clip(visit_momentum - 0.5, 0, 1) +
        0.2 * rng.random(n)
    )
    non_financial_prob = np.clip(
        NON_FINANCIAL_REACTION_RATE * (0.5 + non_financial_score), 0.1, 0.6
    )
    # 全体の30%になるよう閾値調整
    threshold_nf = np.percentile(non_financial_prob, 100 * (1 - NON_FINANCIAL_REACTION_RATE))
    reacted_to_non_financial_incentive = (non_financial_prob >= threshold_nf).astype(int)

    # Step 4: RCTフラグ（非反応者のみ割り当て）
    non_reactor_mask = reacted_to_non_financial_incentive == 0
    dynamic_treatment_flag = np.zeros(n, dtype=int)
    n_non_reactors = non_reactor_mask.sum()
    treatment_assign = rng.random(n_non_reactors) < RCT_TREATMENT_RATIO
    dynamic_treatment_flag[non_reactor_mask] = treatment_assign.astype(int)

    # Step 5: 割引率の提示（Treatment群のみ）
    discount_rate_offered = np.zeros(n)
    treatment_mask = (non_reactor_mask) & (dynamic_treatment_flag == 1)
    discount_rate_offered[treatment_mask] = rng.uniform(
        DISCOUNT_MIN + 0.01, DISCOUNT_MAX, size=treatment_mask.sum()
    ).round(2)

    # ── 7. 離脱確率の生成（is_churn の正解ラベル） ──────────
    # ベース離脱率（ランク別）
    base_churn = np.array([BASE_CHURN_RATE[r] for r in ranks])

    # セグメント調整
    seg_adj = np.array([SEGMENT_CONFIG[s]["base_churn_adj"] for s in segments])
    churn_prob = base_churn + seg_adj

    # 離脱率を上昇させる要因
    churn_prob += np.where(visit_momentum < 0.7, 0.15, 0.0)
    churn_prob += np.where(days_until_next_cycle < 0, 0.10, 0.0)
    churn_prob += np.where(mileage_defense_ratio < 0.6, 0.12, 0.0)
    churn_prob += np.where(past_discount_exposure_count > 5, 0.05, 0.0)

    # 鉄板層の埋め込み（Control群の17%程度に自然離脱率<10%を付与）
    control_mask = (non_reactor_mask) & (dynamic_treatment_flag == 0)
    n_control = control_mask.sum()
    iron_plate_in_control = rng.random(n_control) < IRON_PLATE_RATIO
    iron_plate_indices = np.where(control_mask)[0][iron_plate_in_control]
    churn_prob[iron_plate_indices] = rng.uniform(0.02, 0.09, size=len(iron_plate_indices))

    # 割引効果（Treatment群のみ）
    churn_prob -= (
        discount_rate_offered * price_elasticity * 1.5
    ) * (dynamic_treatment_flag == 1)

    # 非反応者フラグ対象外（非金銭で防衛成功した30%は離脱しない）
    # → is_churn=0 として後で設定
    churn_prob = np.clip(churn_prob, 0.02, 0.95)

    # 離脱ラベルのサンプリング
    is_churn_non_reactor = rng.random(n) < churn_prob

    # 非金銭で防衛成功した顧客は is_churn=0（定義: マイル未達でダウングレード）
    is_churn = np.where(
        reacted_to_non_financial_incentive == 1,
        0,
        is_churn_non_reactor.astype(int)
    )

    # ── 8. DataFrameの組み立て ────────────────────────────
    df = pd.DataFrame({
        "user_id":                           [f"U{i:05d}" for i in range(n)],
        # --- A: 基本・コンテキスト特徴量（モデル入力）
        "lifetime_total_mileage":            lifetime_total_mileage,
        "current_rank":                      ranks,
        "current_rank_encoded":              [RANK_ENCODING[r] for r in ranks],
        "current_90d_mileage":               current_90d_mileage,
        "visit_momentum":                    visit_momentum,
        "avg_spend_recent":                  avg_spend_recent,
        "personal_cycle_days":               personal_cycle_days,
        "days_until_next_cycle":             days_until_next_cycle,
        "past_discount_exposure_count":      past_discount_exposure_count,
        "wishlist_items_count":              wishlist_items_count,
        # --- B: 派生特徴量
        "mileage_defense_ratio":             mileage_defense_ratio.round(3),
        # --- C: 介入・フィルタ・実験フラグ
        "reacted_to_non_financial_incentive": reacted_to_non_financial_incentive,
        "dynamic_treatment_flag":            dynamic_treatment_flag,
        "discount_rate_offered":             discount_rate_offered.round(2),
        # --- D: 配信連携用（モデル非入力）
        "optimal_send_hour":                 optimal_send_hour,
        # --- meta
        "segment":                           segments,
        "price_elasticity":                  price_elasticity.round(2),
        # --- E: 正解ラベル
        "is_churn":                          is_churn,
    })

    return df


def save_data(df: pd.DataFrame, path: str | None = None) -> str:
    """DataFrameをCSVとして保存する。"""
    if path is None:
        path = os.path.join(DATA_DIR, "customers.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"✅ データ保存完了: {path}  ({len(df):,}件)")
    return path


def print_summary(df: pd.DataFrame) -> None:
    """データの基本統計をコンソール出力する。"""
    print("\n" + "=" * 60)
    print("📊 合成データ生成サマリー")
    print("=" * 60)

    total = len(df)
    reactors   = df["reacted_to_non_financial_incentive"].sum()
    non_react  = total - reactors
    treatment  = df["dynamic_treatment_flag"].sum()
    control    = non_react - treatment

    print(f"\n  総顧客数:              {total:,}人")
    print(f"  非金銭インセンティブ反応: {reactors:,}人 ({reactors/total:.1%})")
    print(f"    └ コントロール群:      {control:,}人 ({control/total:.1%})")
    print(f"    └ トリートメント群:    {treatment:,}人 ({treatment/total:.1%})")
    print(f"\n  全体離脱率:            {df['is_churn'].mean():.1%}")
    print(f"  非反応者の離脱率:       {df[df['reacted_to_non_financial_incentive']==0]['is_churn'].mean():.1%}")

    print("\n【ランク別分布】")
    rank_stats = df.groupby("current_rank").agg(
        顧客数=("user_id", "count"),
        離脱率=("is_churn", "mean"),
        平均客単価=("avg_spend_recent", "mean"),
        平均90日マイル=("current_90d_mileage", "mean"),
    ).round(2)
    print(rank_stats.to_string())

    print("\n【セグメント別分布】")
    seg_stats = df.groupby("segment").agg(
        顧客数=("user_id", "count"),
        離脱率=("is_churn", "mean"),
        平均客単価=("avg_spend_recent", "mean"),
        平均来店周期=("personal_cycle_days", "mean"),
    ).round(2)
    print(seg_stats.to_string())
    print()


if __name__ == "__main__":
    df = generate_data()
    save_data(df)
    print_summary(df)
