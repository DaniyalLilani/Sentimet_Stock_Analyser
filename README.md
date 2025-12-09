# 📈 News-Driven Stock Movement Prediction with FinBERT & Deep Learning

## 🔍 Project Overview

This project is an **AI-powered stock movement prediction system** that uses:

- **Financial news headlines**
- **FinBERT embeddings for NLP sentiment understanding**
- **Market & fundamental numerical features**
- **A PyTorch-based Multilayer Perceptron (MLP)**

The model predicts the **direction and magnitude of stock price movement** shortly after a news event.

The goal is to simulate **real-world trading-time inference**, where only information available at the time of news release is used — avoiding any form of data leakage.

---
## How to run [IMPORTANT!]
Running FINbert is computationally very expensive and without CUDA can take many hours. We have therefore provided an ```after_fin_finance.csv``` that has the finbert sentiment values from a previous run along with encoded ```stock_name``` ```sector``` and ```industry```
The notebook has been set up in a way for you to run from Step 6
Please use the following Link https://drive.google.com/file/d/1CB1DHfaOrY_9MYbxR-6kCGBXgBnBYAv5/view?usp=sharing to download after_fin_finance.
For reference, finance.csv is avalible here as well (pre FINbert) https://drive.google.com/file/d/11XG0r3RdwNHESDqUwZvZGK5W3SNETg1P/view?usp=sharing


## 🎯 Prediction Objective

The model performs **5-class classification** on stock price movement:

| Class | Meaning |
|-------|---------|
| 1 | Small Down |
| 2 | Medium Down |
| 3 | Neutral / Slight Up |
| 4 | Medium Up |
| 5 | Large Up |

Additionally, a **direction-only metric** is tracked:

- **Down:** Classes 1–2  
- **Up:** Classes 3–5  

This allows evaluation of both:
- **Exact accuracy**
- **Directional trading accuracy**

---

## 🧠 Model Architecture

The core model is a **deep MLP classifier**:

Input → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
→ Linear(128) → BatchNorm → ReLU → Dropout(0.2)
→ Linear(5 outputs)


- **Optimizer:** AdamW  
- **Loss Function:** CrossEntropyLoss  
- **Batch Size:** 512  
- **Learning Rate:** 3e-4  
- **Weight Decay:** 5e-5  

---

## 📊 Input Features

### ✅ NLP Features (FinBERT)
- `finbert_emb_0 ... finbert_emb_127`

These embeddings capture the contextual financial sentiment of each headline.

### ✅ Sentiment & News Flags
- `sent_neg`, `sent_neu`, `sent_pos`
- `has_fda`, `has_merger`, `has_upgrade`, `has_downgrade`

### ✅ Market & Fundamental Features (Scaled)
- `volume`
- `open_price`, `high_price`, `low_price`, `close_price`
- `float_shares`, `shares_outstanding`
- `market_cap`
- `pe_ratio`
- `daily_volatility_pct`

### ✅ Encoded Categorical Features
- `stock_name_encoded`
- `sector_encoded`
- `industry_encoded`

❌ **Explicitly excluded for leakage prevention:**
- `signed_price_move_pct`
- Any future-looking market data

---

## 🔒 Data Leakage Prevention

Strict steps are taken to ensure real-world validity:

- Train/Validation split is performed **before scaling**
- `StandardScaler` is fit only on the **training set**
- `signed_price_move_pct` is **never used** as a feature
- Deployment pipeline only uses **time-of-news features**

A dedicated `deployment_features` list ensures that only safe features are used during inference.

---

## 🏋️ Training Process

During training, the following metrics are tracked:

- ✅ Training & Validation Loss
- ✅ Exact 5-Class Validation Accuracy
- ✅ Directional Accuracy (Up vs Down)
- ✅ Training & Validation Confidence Scores

The **best model is selected automatically** using validation loss and saved to:
best_mlp_checkpoint.pth

## 🚀 Mock Deployment System

A full **mock deployment pipeline** simulates real-world usage:

- Random unseen validation samples are selected
- Only safe, time-of-news features are used
- The model outputs:
  - Predicted movement class
  - Confidence score
  - Direction prediction
- Results are compared against actual outcomes

### Contributors
Daniyal Lilani (100867494)
Aryan Khashefi-Aazam
Abullah Mustafa

