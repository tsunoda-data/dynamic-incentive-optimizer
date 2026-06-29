"""
train_uplift.py — DIO-KFC v2: T-Learner (Uplift Modeling)

2つの独立したLightGBMモデルを学習する:
  - Model A (Control):   dynamic_treatment_flag=0 で学習
                         出力: 自然離脱確率 P(churn | d=0)
  - Model B (Treatment): dynamic_treatment_flag=1 で学習、discount_rate_offered を含む
                         出力: 介入時離脱確率 P(churn | d)

学習対象: reacted_to_non_financial_incentive=0 のサブセットのみ（約7,000人）

評価指標:
  - AUC-ROC / PR-AUC / Brier Score / キャリブレーション曲線
  - Qini曲線 & Qini係数 / AUUC / Uplift by Decile
  - SHAP Summary Plot (Model A vs Model B 対比)
  - SHAP Waterfall Plot (3パターン: 鉄板層/説得層/高弾力説得層)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss
)

from config import (
    RANDOM_SEED, OUTPUT_DIR, DATA_DIR,
    LGBM_PARAMS, LGBM_NUM_BOOST_ROUND, LGBM_EARLY_STOPPING,
    IRON_PLATE_THRESHOLD, FIGURE_DPI, KFC_RED,
)

import os

# ─── 特徴量リスト ──────────────────────────────────────────────
FEATURES_BASE = [
    "lifetime_total_mileage",
    "current_rank_encoded",
    "current_90d_mileage",
    "mileage_defense_ratio",
    "visit_momentum",
    "avg_spend_recent",
    "personal_cycle_days",
    "days_until_next_cycle",
    "past_discount_exposure_count",
    "wishlist_items_count",
]
# Model B のみ追加する操作変数
DISCOUNT_FEATURE = "discount_rate_offered"
FEATURES_TREATMENT = FEATURES_BASE + [DISCOUNT_FEATURE]

TARGET = "is_churn"


# ─── ユーティリティ ───────────────────────────────────────────

def _train_lgbm(X_train, y_train, X_val, y_val, params, num_rounds, early_stopping):
    """LightGBMモデルを学習して返す。"""
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)
    callbacks = [lgb.early_stopping(early_stopping, verbose=False),
                 lgb.log_evaluation(period=-1)]
    model = lgb.train(
        params, dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


def _evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """AUC-ROC / PR-AUC / Brier Score を計算して返す。"""
    y_prob = model.predict(X_test)
    auc    = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier  = brier_score_loss(y_test, y_prob)
    print(f"  [{model_name}] AUC-ROC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
    return {"model_name": model_name, "auc": auc, "pr_auc": pr_auc, "brier": brier,
            "y_prob": y_prob, "y_test": y_test}


def compute_qini(cate: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> dict:
    """
    Qini曲線とQini係数を計算する。

    Qini係数 = Upliftモデルの曲線下面積 − ランダム施策の面積

    Parameters
    ----------
    cate      : 各顧客のCATE推定値（高いほど施策効果が高い）
    y         : 正解ラベル (is_churn)
    treatment : 処置フラグ (dynamic_treatment_flag)
    """
    n = len(cate)
    # CATEスコアの降順に並べる
    order = np.argsort(-cate)
    y_ord = y[order]
    t_ord = treatment[order]

    n_t = treatment.sum()
    n_c = (1 - treatment).sum()
    overall_uplift = (
        y_ord[t_ord == 1].mean() - y_ord[t_ord == 0].mean()
    ) if n_t > 0 and n_c > 0 else 0

    # 累積Uplift（割合ごと）
    proportions, uplifts = [], []
    for k in range(1, n + 1):
        sub_y = y_ord[:k]
        sub_t = t_ord[:k]
        n_tk = sub_t.sum()
        n_ck = k - n_tk
        if n_tk > 0 and n_ck > 0:
            u = sub_y[sub_t == 1].mean() - sub_y[sub_t == 0].mean()
        else:
            u = 0.0
        proportions.append(k / n)
        uplifts.append(u)

    proportions = np.array(proportions)
    uplifts     = np.array(uplifts)

    # ランダム施策の直線
    random_line = np.linspace(0, overall_uplift, n)

    # Qini係数 = モデル曲線 − ランダム直線 の面積
    qini_coef = np.trapz(uplifts - random_line, proportions)

    return {
        "proportions":   proportions,
        "uplifts":       uplifts,
        "random_line":   random_line,
        "qini_coef":     qini_coef,
        "overall_uplift": overall_uplift,
    }


def compute_uplift_by_decile(cate: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> pd.DataFrame:
    """Uplift by Decile テーブルを計算する。"""
    df = pd.DataFrame({"cate": cate, "y": y, "treatment": treatment})
    df["decile"] = pd.qcut(-df["cate"], q=10, labels=False) + 1  # 1=最高CATE

    rows = []
    for d in range(1, 11):
        sub = df[df["decile"] == d]
        n_t = (sub["treatment"] == 1).sum()
        n_c = (sub["treatment"] == 0).sum()
        if n_t > 0 and n_c > 0:
            uplift = sub[sub["treatment"] == 1]["y"].mean() - sub[sub["treatment"] == 0]["y"].mean()
        else:
            uplift = np.nan
        rows.append({"Decile": d, "N": len(sub), "N_Treatment": n_t, "N_Control": n_c,
                     "Uplift": uplift})
    return pd.DataFrame(rows)


# ─── メインパイプライン ────────────────────────────────────────

def run_training_pipeline(df: pd.DataFrame | None = None):
    """
    T-Learnerを学習し、モデル・評価指標・SHAPを返す。

    Returns
    -------
    model_a, model_b : lgb.Booster
    df_result        : 元のdfに自然離脱確率・CATE等を付与したDataFrame
    metrics          : 評価指標dict
    """
    if df is None:
        from generate_data import generate_data
        df = generate_data()

    print("\n" + "=" * 60)
    print("🤖 Stage 1: Uplift Modeling (T-Learner)")
    print("=" * 60)

    # ── 学習対象: 非反応者のみ ──────────────────────────────
    df_uplift = df[df["reacted_to_non_financial_incentive"] == 0].copy().reset_index(drop=True)
    print(f"  T-Learner 学習対象: {len(df_uplift):,}人"
          f" (全体の {len(df_uplift)/len(df):.1%})")

    df_control   = df_uplift[df_uplift["dynamic_treatment_flag"] == 0].reset_index(drop=True)
    df_treatment = df_uplift[df_uplift["dynamic_treatment_flag"] == 1].reset_index(drop=True)
    print(f"  Control群:   {len(df_control):,}人")
    print(f"  Treatment群: {len(df_treatment):,}人")

    # ── Model A: コントロール ────────────────────────────────
    print("\n  [Model A — Control] 学習中...")
    Xc = df_control[FEATURES_BASE]
    yc = df_control[TARGET]
    Xc_tr, Xc_val, yc_tr, yc_val = train_test_split(
        Xc, yc, test_size=0.2, random_state=RANDOM_SEED, stratify=yc
    )
    model_a = _train_lgbm(Xc_tr, yc_tr, Xc_val, yc_val,
                          LGBM_PARAMS, LGBM_NUM_BOOST_ROUND, LGBM_EARLY_STOPPING)
    metrics_a = _evaluate_model(model_a, Xc_val, yc_val, "Model A (Control)")

    # ── Model B: トリートメント ──────────────────────────────
    print("\n  [Model B — Treatment] 学習中...")
    Xt = df_treatment[FEATURES_TREATMENT]
    yt = df_treatment[TARGET]
    Xt_tr, Xt_val, yt_tr, yt_val = train_test_split(
        Xt, yt, test_size=0.2, random_state=RANDOM_SEED, stratify=yt
    )
    model_b = _train_lgbm(Xt_tr, yt_tr, Xt_val, yt_val,
                          LGBM_PARAMS, LGBM_NUM_BOOST_ROUND, LGBM_EARLY_STOPPING)
    metrics_b = _evaluate_model(model_b, Xt_val, yt_val, "Model B (Treatment)")

    # ── CATE 推定（全非反応者） ─────────────────────────────
    # Model A: 自然離脱確率
    p_control_all  = model_a.predict(df_uplift[FEATURES_BASE])
    # Model B: 介入時離脱確率（割引率は実際の提示値を使用、Controlは0.00）
    Xt_all = df_uplift[FEATURES_TREATMENT].copy()
    p_treatment_all = model_b.predict(Xt_all)

    # CATE(d) = P_A - P_B(d)
    cate_all = p_control_all - p_treatment_all
    df_uplift = df_uplift.assign(
        p_natural_churn=p_control_all,
        p_treated_churn=p_treatment_all,
        cate=cate_all,
    )

    # ── Qini曲線 ──────────────────────────────────────────
    print("\n  Qini曲線を計算中...")
    qini = compute_qini(
        cate_all,
        df_uplift[TARGET].values,
        df_uplift["dynamic_treatment_flag"].values,
    )
    print(f"  Qini係数: {qini['qini_coef']:.4f}")

    uplift_decile = compute_uplift_by_decile(
        cate_all,
        df_uplift[TARGET].values,
        df_uplift["dynamic_treatment_flag"].values,
    )
    print("\n  Uplift by Decile:")
    print(uplift_decile.to_string(index=False))

    # ── SHAP分析 ─────────────────────────────────────────
    print("\n  SHAP分析中...")
    _plot_shap_summary(model_a, Xc_tr, "Model A (Control)", "shap_summary_model_a.png")
    _plot_shap_summary(model_b, Xt_tr, "Model B (Treatment)", "shap_summary_model_b.png")
    _plot_shap_waterfall(model_a, model_b, df_uplift)
    _plot_calibration(metrics_a, metrics_b)

    metrics = {
        "model_a": metrics_a,
        "model_b": metrics_b,
        "qini":    qini,
        "uplift_by_decile": uplift_decile,
    }

    # df_uplift を元のdfにマージして返す
    df_out = df.copy()
    df_out["p_natural_churn"] = np.nan
    df_out["p_treated_churn"] = np.nan
    df_out["cate"]            = np.nan
    non_react_idx = df["reacted_to_non_financial_incentive"] == 0
    df_out.loc[non_react_idx, "p_natural_churn"] = df_uplift["p_natural_churn"].values
    df_out.loc[non_react_idx, "p_treated_churn"] = df_uplift["p_treated_churn"].values
    df_out.loc[non_react_idx, "cate"]            = df_uplift["cate"].values

    print("\n✅ Stage 1 完了\n")
    return model_a, model_b, df_out, metrics, qini


def _plot_shap_summary(model, X_train, title: str, filename: str):
    """SHAP Summary Plot を生成・保存する。"""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_train)

    shap.summary_plot(shap_vals, X_train, show=False,
                      plot_type="dot", max_display=11)
    fig = plt.gcf()
    plt.title(f"SHAP Summary — {title}", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    保存: {path}")


def _plot_shap_waterfall(model_a, model_b, df_uplift: pd.DataFrame):
    """
    SHAP Waterfall Plot を3パターン生成する。
    パターンA: 鉄板層（p_natural_churn < 0.10）
    パターンB: 中程度説得層
    パターンC: 高弾力説得層（高CATE）
    """
    patterns = [
        ("A_鉄板層",     df_uplift[df_uplift["p_natural_churn"] < IRON_PLATE_THRESHOLD]),
        ("B_中程度説得層", df_uplift[(df_uplift["cate"] > 0.05) & (df_uplift["cate"] <= 0.20)]),
        ("C_高弾力説得層", df_uplift[df_uplift["cate"] > 0.20]),
    ]

    explainer_a = shap.TreeExplainer(model_a)

    for label, subset in patterns:
        if len(subset) == 0:
            continue
        sample   = subset.sample(1, random_state=RANDOM_SEED).reset_index(drop=True)
        X_sample = sample[FEATURES_BASE]
        sv       = explainer_a(X_sample)

        shap.waterfall_plot(sv[0], show=False, max_display=11)
        fig = plt.gcf()
        fig.suptitle(f"SHAP Waterfall — パターン {label}", fontsize=12,
                     fontweight="bold", y=1.01)
        path = os.path.join(OUTPUT_DIR, f"shap_waterfall_{label}.png")
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"    保存: {path}")


def _plot_calibration(metrics_a: dict, metrics_b: dict):
    """キャリブレーション曲線を2モデル対比で描画する。"""
    fig, ax = plt.subplots(figsize=(7, 5))

    for m, color, label in [
        (metrics_a, KFC_RED,   "Model A (Control)"),
        (metrics_b, "#457B9D", "Model B (Treatment)"),
    ]:
        fraction_pos, mean_pred = calibration_curve(
            m["y_test"], m["y_prob"], n_bins=10
        )
        ax.plot(mean_pred, fraction_pos, marker="o", color=color,
                label=f"{label}\n  Brier={m['brier']:.4f}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_xlabel("予測離脱確率")
    ax.set_ylabel("実際の離脱率")
    ax.set_title("キャリブレーション曲線 — Model A vs Model B", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "calibration_curves.png")
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"    保存: {path}")


if __name__ == "__main__":
    from generate_data import generate_data, save_data
    df = generate_data()
    save_data(df)
    model_a, model_b, df_out, metrics, qini = run_training_pipeline(df)
