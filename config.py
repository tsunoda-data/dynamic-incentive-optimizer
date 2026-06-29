"""
config.py — DIO-KFC v2: Dynamic Incentive Optimizer
全定数・ハイパーパラメータの一元管理

設計思想:
  - 公式ルール(表): 90日ローリング判定 + 1段階ダウングレード
  - AI検知(裏): 個人サイクル監視 → 完全デジタル非金銭 → CATE最適化割引
"""

import os

# ─────────────────────────────────────────────
# 1. プロジェクトパス
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. 乱数シード（全モジュール共通）
# ─────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 3. ランク体系（90日ローリング判定）
# ─────────────────────────────────────────────
RANKS = ["レギュラー", "ブロンズ", "シルバー", "ゴールド", "プラチナ"]

# 90日間に必要なマイル数（v1の30日基準から変更）
RANK_THRESHOLDS = {
    "レギュラー": 0,
    "ブロンズ":   30,
    "シルバー":   100,
    "ゴールド":   200,
    "プラチナ":   400,
}

# ランクの序数エンコーディング（0〜4）
RANK_ENCODING = {rank: i for i, rank in enumerate(RANKS)}
RANK_DECODING = {i: rank for i, rank in enumerate(RANKS)}

# ランク別構成比（合成データ生成用）
RANK_DISTRIBUTION = {
    "レギュラー": 0.40,
    "ブロンズ":   0.25,
    "シルバー":   0.20,
    "ゴールド":   0.10,
    "プラチナ":   0.05,
}

# ─────────────────────────────────────────────
# 4. 顧客セグメント定義
# ─────────────────────────────────────────────
SEGMENTS = ["ファミリー", "単身ヘビー", "学生", "シニア"]

SEGMENT_CONFIG = {
    "ファミリー": {
        "ratio":           0.35,
        "spend_range":     (1800, 3500),
        "cycle_range":     (14, 21),    # 来店周期(日)
        "price_elasticity": 0.4,        # 低弾力性（割引効果が小さい）
        "base_churn_adj":  -0.03,       # ランクベース離脱率への調整
    },
    "単身ヘビー": {
        "ratio":           0.20,
        "spend_range":     (800, 1500),
        "cycle_range":     (7, 14),
        "price_elasticity": 0.6,
        "base_churn_adj":  -0.02,
    },
    "学生": {
        "ratio":           0.25,
        "spend_range":     (500, 1000),
        "cycle_range":     (21, 45),
        "price_elasticity": 0.9,        # 高弾力性（割引効果が大きい）
        "base_churn_adj":  +0.03,
    },
    "シニア": {
        "ratio":           0.20,
        "spend_range":     (700, 1200),
        "cycle_range":     (30, 60),
        "price_elasticity": 0.3,
        "base_churn_adj":  +0.02,
    },
}

# ─────────────────────────────────────────────
# 5. 合成データ生成パラメータ
# ─────────────────────────────────────────────
N_CUSTOMERS = 10_000

# 非金銭インセンティブの反応率
# 完全デジタルコンテンツ（バッジ/先行情報/デジタルスタンプ）への反応
NON_FINANCIAL_REACTION_RATE = 0.30  # 全体の30%がコストゼロで防衛成功

# RCTフラグの割り当て比率（非反応者のみ）
RCT_TREATMENT_RATIO = 0.50  # Control: 50% / Treatment: 50%

# ランク別ベース離脱率（割引なし・非反応者）
BASE_CHURN_RATE = {
    "レギュラー": 0.30,
    "ブロンズ":   0.25,
    "シルバー":   0.20,
    "ゴールド":   0.15,
    "プラチナ":   0.12,
}

# 鉄板層の埋め込み比率（Control群の中に自然離脱率<10%の顧客を意図的に含める）
IRON_PLATE_RATIO = 0.17  # Control群の約17%

# ─────────────────────────────────────────────
# 6. ビジネスパラメータ（最適化）
# ─────────────────────────────────────────────
COST_RATE = 0.40    # 原価率 C

# 割引率グリッド（操作変数）
DISCOUNT_MIN  = 0.00
DISCOUNT_MAX  = 0.30
DISCOUNT_STEP = 0.01

# ペナルティ項: Penalty(d) = λ × V × d² × past_discount_exposure_count
PENALTY_LAMBDA = 1.5  # デフォルト値（感度分析対象）
PENALTY_LAMBDA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]  # 感度分析用

# 鉄板層閾値: P_A(churn) < IRON_PLATE_THRESHOLD → d* = 0 を強制
IRON_PLATE_THRESHOLD = 0.10

# ベースライン比較用の一律割引率
UNIFORM_DISCOUNT_RATE = 0.20

# ─────────────────────────────────────────────
# 7. LightGBMハイパーパラメータ
# ─────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":          -1,
    "random_state":     RANDOM_SEED,
}

LGBM_NUM_BOOST_ROUND = 500
LGBM_EARLY_STOPPING  = 50

# ─────────────────────────────────────────────
# 8. 非金銭インセンティブ定義（完全デジタル完結）
# ─────────────────────────────────────────────
# 物理オペレーションを一切伴わない、サーバーサイド配信完結のコンテンツのみ
DIGITAL_INCENTIVES = [
    "シークレット・カーネルバッジ（称号）",
    "次回新商品の先行情報・ビジュアルへの限定アクセス権",
    "オリジナルスマホ壁紙・限定デジタルスタンプの配布",
]

# ─────────────────────────────────────────────
# 9. 可視化設定
# ─────────────────────────────────────────────
# KFCブランドカラー
KFC_RED    = "#C8102E"
KFC_WHITE  = "#FFFFFF"
KFC_CREAM  = "#F5E6C8"

# セグメント別カラーパレット
SEGMENT_COLORS = {
    "ファミリー": "#E63946",
    "単身ヘビー": "#457B9D",
    "学生":       "#2A9D8F",
    "シニア":     "#E9C46A",
}

# 顧客タイプ別カラー（Uplift Modeling用）
CATE_COLORS = {
    "説得層 (Persuadables)": "#2A9D8F",
    "鉄板層 (Sure Things)":  "#457B9D",
    "天邪鬼層 (Sleeping Dogs)": "#E63946",
}

RANK_COLORS = {
    "レギュラー": "#8D8D8D",
    "ブロンズ":   "#CD7F32",
    "シルバー":   "#A8A9AD",
    "ゴールド":   "#FFD700",
    "プラチナ":   "#E5E4E2",
}

FIGURE_DPI = 150
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
