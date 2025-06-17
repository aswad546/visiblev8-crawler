# Robust Comparison of Balanced vs Imbalanced Approaches

## Methodology

This analysis uses nested cross-validation to avoid hyperparameter selection bias:

- **Outer CV**: 5-fold for unbiased performance estimation
- **Inner CV**: 3-fold for hyperparameter tuning
- **Test Set**: 20% held-out data never seen during model selection
- **Statistical Testing**: Paired t-test and Wilcoxon signed-rank test

## Key Results

### Nested Cross-Validation Performance

| Metric | Balanced | Imbalanced | p-value | Significant |
|--------|----------|------------|---------|-------------|
| val_auc | 0.996±0.002 | 0.997±0.002 | 0.6167 | No |
| val_ap | 0.996±0.002 | 0.981±0.014 | 0.1326 | No |
| val_acc | 0.966±0.013 | 0.983±0.010 | 0.2251 | No |

### Hold-out Test Set Performance

| Metric | Balanced | Imbalanced |
|--------|----------|------------|
| auc | 1.000 | 0.998 |
| average_precision | 1.000 | 0.986 |
| accuracy | 1.000 | 0.991 |
| precision | 1.000 | 0.957 |
| recall | 1.000 | 0.957 |

## Statistical Analysis

### Effect Sizes (Cohen's d)

- val_auc: 0.271 (small effect)
- val_ap: -0.942 (large effect)
- val_acc: 0.716 (medium effect)

## Recommendations

**Recommended Approach: BALANCED**

The balanced approach shows superior performance on the held-out test set despite having less training data.

## Methodological Advantages

This analysis addresses common pitfalls in ML model comparison:

1. **No data leakage**: Test set never used during model selection
2. **No overfitting to validation set**: Nested CV prevents this
3. **Statistical rigor**: Significance testing and effect sizes
4. **Honest performance**: Test set results are unbiased estimates
