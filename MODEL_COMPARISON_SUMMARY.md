# Model Comparison — Quick Reference

## 🏆 Winner: Isolation Forest

**Performance:** AUC-ROC 0.701 | Avg Precision 0.530 | Weighted Score 0.564

---

## 📊 Full Rankings

```
┌─────┬──────────────────────┬─────────┬───────────────┬────────────┬────────────────┐
│ Rank│ Model                │ AUC-ROC │ Avg Precision │ Silhouette │ Weighted Score │
├─────┼──────────────────────┼─────────┼───────────────┼────────────┼────────────────┤
│  1  │ Isolation Forest     │  0.701  │     0.530     │   0.273    │     0.564      │
│  2  │ Ensemble (IsoF+KDE)  │  0.687  │     0.496     │   0.261    │     0.545      │
│  3  │ Spectral Clustering  │  0.673  │     0.475     │   0.269    │     0.533      │
│  4  │ KDE                  │  0.675  │     0.477     │   0.260    │     0.532      │
│  5  │ Gaussian Mixture     │  0.644  │     0.456     │   0.223    │     0.503      │
│  6  │ HDBSCAN              │  0.520  │     0.356     │   0.200    │     0.407      │
│  7  │ DBSCAN               │  0.500  │     0.311     │   0.000    │     0.343      │
└─────┴──────────────────────┴─────────┴───────────────┴────────────┴────────────────┘
```

---

## 🎯 Key Insights

### Top 3 Models
1. **Isolation Forest** — Best anomaly detection, highest AUC-ROC
2. **Ensemble** — Robust combination, close second
3. **Spectral** — Best clustering approach, good spatial capture

### Bottom 3 Models
5. **Gaussian Mixture** — Moderate performance, probabilistic approach
6. **HDBSCAN** — Struggled with district-level data
7. **DBSCAN** — Poor performance, too sensitive to density

---

## ✅ Recommendation

**Deploy Isolation Forest as primary model**

**Why:**
- 3.5% better than current ensemble
- Simpler architecture
- Faster prediction
- Easier to maintain

**Alternative:** Keep ensemble for robustness (acceptable trade-off)

---

## 📁 Files

- **Full Report:** `reports/model_comparison/model_comparison_report.txt`
- **Results CSV:** `reports/model_comparison/model_comparison_results.csv`
- **Visualizations:** `reports/model_comparison/*.png`
- **Complete Summary:** `MODEL_COMPARISON_COMPLETE.md`

---

## 🔢 Metrics Explained

- **AUC-ROC:** How well model separates high/low risk (higher = better)
- **Avg Precision:** Precision-recall balance (higher = better)
- **Silhouette:** Clustering quality (higher = better)
- **Weighted Score:** 50% AUC + 30% Precision + 20% Silhouette

---

**Study Status:** ✅ COMPLETE
**Best Model:** Isolation Forest
**Performance Gain:** +3.5% over current ensemble
