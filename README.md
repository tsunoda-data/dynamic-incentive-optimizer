# Dynamic Incentive Optimizer for QSR Loyalty Program

**Applying Duolingo's Streak Psychology to KFC's Chicken Miles — Uplift Modeling × Mathematical Optimization**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-T--Learner-9ACD32)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6F61)](https://shap.readthedocs.io/)
[![Colab](https://img.shields.io/badge/Google%20Colab-Run%20Now-F9AB00?logo=googlecolab)](notebooks/full_pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

> **For Business & Marketing Context**  
> この分析のビジネス背景・CRM戦略についての詳細は、NOTE記事をご覧ください。  
> [NOTE記事：「毎月28日しか行かない」顧客の脳内をハックせよ]()

---

## Overview

クイックサービスレストラン（QSR）の会員プログラムにおける**顧客離脱を防ぎ、LTV（顧客生涯価値）を最大化**する、Uplift Modeling × 数理最適化のインセンティブ制御システム。

### 概要

| 項目 | 内容 |
|------|-----|
| アーキテクチャ | **T-Learner (Uplift Modeling)** |
| ランク判定期間 | **90日ローリング** |
| 非金銭インセンティブ | **完全デジタル完結** |
| 目的関数 | **CATE × 粗利 − 割引コスト − ペナルティ** |
| 評価指標 | **Qini曲線, CATE分布, 増分ROI** |

### Core Idea: Duolingo → KFC Transfer

| Duolingo | Behavioral Economics | KFC Application |
|----------|---------------------|-----------------|
| Streak (consecutive days) | **Loss Aversion** | 90-day rolling rank defense |
| Streak Freeze (grace) | **Recovery Design** | Digital-only incentive → discount escalation |
| League ranking | **Endowment Effect** | Rank demotion alert |
| Goal gradient | **Goal Gradient Effect** | `personal_cycle_days` cycle management |
| Partial reinforcement | **Operant Conditioning** | Non-monetary → monetary stage control |

---

## Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  DATA GENERATION LAYER                       │
│  generate_data.py                                            │
│                                                              │
│  10,000 Customers × 4 Segments × Seasonal Effects           │
│  → 段階的介入フロー:                                           │
│     全顧客 → 非金銭インセンティブ(30%反応=コストゼロ防衛)          │
│           → 非反応者(70%) → RCT(50:50) → T-Learner対象      │
├─────────────────────────────────────────────────────────────┤
│             STAGE 1: UPLIFT MODELING (T-Learner)             │
│  train_uplift.py                                             │
│                                                              │
│  Model A (Control):   P_A = P(churn | d=0)                  │
│  Model B (Treatment): P_B = P(churn | d)  ← d を含む        │
│  CATE(d) = P_A - P_B(d)                                      │
│                                                              │
│  評価: AUC / PR-AUC / Brier / Qini曲線 / SHAP×2モデル        │
├─────────────────────────────────────────────────────────────┤
│         STAGE 2: CATE-BASED PROFIT MAXIMIZATION              │
│  optimize_incentive.py                                       │
│                                                              │
│  Profit(d) = CATE(d)×V×(1-C) - d×V×(1-P_B) - λVd²E        │
│                                                              │
│  鉄板層 [P_A < 0.10] → d* = 0 自動強制                        │
│  Grid Search: d = 0.00, 0.01, ..., 0.30                     │
├─────────────────────────────────────────────────────────────┤
│              EVALUATION & VISUALIZATION                      │
│  evaluate_roi.py + visualize.py                              │
│                                                              │
│  Qini Curve | CATE分布 | 増分ROI | 無駄削減率 | λ感度分析      │
└─────────────────────────────────────────────────────────────┘
```

---

## なぜT-Learnerを使うのか？（目的関数への直結）

単なる「離脱確率予測＋グリッドサーチ」との本質的な違いは、**CATEを最適化の中核に据える**点にある。

$$\text{Profit}_i(d) = \underbrace{\text{CATE}_i(d) \times V_i \times (1-C)}_{\text{増分粗利（T-Learnerの価値）}} - \underbrace{d \times V_i \times (1-P_B)}_{\text{割引コスト}} - \underbrace{\lambda V_i d^2 E_i}_{\text{割引慣れペナルティ}}$$

> **鉄板層では CATE ≈ 0 → 第1項 ≈ 0 → 割引コストが利益を必ず上回る → $d^* = 0$ が自動選択**  
> T-Learnerを構築した必然性が最適化ロジック本体に直結している。

---

## Rank System（KFC チキンマイル準拠 / 90日判定）

| Rank | 90日間必要マイル | ダウングレードルール |
|------|----------------|----------------|
| レギュラー | 0 | — |
| ブロンズ | 30 | 未達 → レギュラーへ1段階のみ |
| シルバー | 100 | 未達 → ブロンズへ1段階のみ |
| ゴールド | 200 | 未達 → シルバーへ1段階のみ |
| プラチナ | 400 | 未達 → ゴールドへ1段階のみ |

---

## Non-Financial Incentives（完全デジタル完結）

> QSRのピークタイムにおける「優先確保」「裏メニュー」は **限界費用ゼロどころか甚大な機会損失**を生む。  
> 本システムでは、**店舗オペレーションへの負荷がゼロのデジタルコンテンツのみ**を使用する。

- アプリ内限定シークレット・カーネルバッジ（称号）
- 次回新商品の先行情報・ビジュアルへの限定アクセス権
- オリジナルスマホ壁紙・限定デジタルスタンプの配布

---

## Visualization Outputs

| # | 図表 | 優先度 | 目的 |
|---|------|--------|------|
| 1 | Qini曲線 | **A** | T-Learner vs ランダム施策の定量比較 |
| 2 | 割引率 vs 利益カーブ（ペナルティあり/なし） | **A** | ペナルティの抑制効果の可視化 |
| 3 | CATE分布ヒストグラム | **A** | 鉄板層/説得層の分離度証明 |
| 4 | SHAP Waterfall Plot（3パターン） | **A** | 個別最適化の説明可能性 |
| 5 | ROI比較（一律20% vs AI最適化） | B | 増分利益の定量証明 |
| 6 | 無駄な割引カット率（ランク別） | B | 鉄板層への不要コスト削減 |
| 7 | SHAP Summary（Model A / Model B 対比） | B | 2モデルの特徴量重要度の差 |
| 8 | ペナルティ強度λの感度分析 | B | λ変化への利益応答 |
| 9 | キャリブレーション曲線 | B | 確率の信頼性検証 |

---

## Repository Structure

```
dynamic-incentive-optimizer-v2/
├── README.md
├── requirements.txt
├── config.py              # 定数・ハイパーパラメータ（ランク閾値/λ/鉄板層閾値等）
├── generate_data.py       # 合成データ生成（10K×4セグメント×段階的介入）
├── train_uplift.py        # T-Learner学習・Qini・SHAP
├── optimize_incentive.py  # CATEベース期待利益最大化・感度分析
├── evaluate_roi.py        # 増分ROI・CATE分布・Qini描画
├── visualize.py           # 利益カーブ・λ感度分析・割引分布
├── notebooks/
│   └── full_pipeline.ipynb  # Google Colab 一括実行
└── outputs/               # 生成される全図表
```

---

## Quick Start

### Option 1: Google Colab（推奨）
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/full_pipeline.ipynb)

「ランタイム → すべてのセルを実行」で一括実行できます。

### Option 2: Local Execution

```bash
git clone https://github.com/YOUR_USERNAME/dynamic-incentive-optimizer-v2.git
cd dynamic-incentive-optimizer-v2
pip install -r requirements.txt

python generate_data.py       # → data/customers.csv
python train_uplift.py        # → Model A / Model B + SHAP + Qini
python optimize_incentive.py  # → 最適割引率 d*
python evaluate_roi.py        # → 増分ROI + CATE分布
python visualize.py           # → outputs/*.png
```

---

## Implementation Roadmap

| Phase | Content | Status |
|-------|---------|--------|
| **Phase 0** | 引当金PL影響シミュレーション + グランドファーザー条項 | 📋 Prerequisite |
| **Phase 1** | 動的ホールドアウトA/BテストによるRCTデータ蓄積 | 🔜 Planned |
| **Phase 2** | 完全デジタル非金銭インセンティブの本番展開 | 🔜 Planned |
| **Phase 3** | CATEベースAI最適化の全面展開 |  **本リポジトリで実装** |

---

## Ethics & Data Privacy

> 本プロジェクトで使用するデータは **100% 合成データ（Synthetic Data）** です。  
> 実際の顧客データは一切使用していません。KFCの公開情報を参考にしたケーススタディであり、日本ケンタッキー・フライド・チキン株式会社との提携・委託関係はありません。

---

## Author

**[Your Name]** — CRM Marketer × Data Scientist

-  [NOTE]()
-  [GitHub]()
-  [LinkedIn]()
