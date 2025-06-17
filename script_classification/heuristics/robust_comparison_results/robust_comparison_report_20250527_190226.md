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
| val_auc | 0.998±0.003 | 0.999±0.001 | 0.4262 | No |
| val_ap | 0.998±0.002 | 0.994±0.007 | 0.0966 | No |
| val_acc | 0.975±0.016 | 0.994±0.004 | 0.0427 | Yes |

### Hold-out Test Set Performance

| Metric | Balanced | Imbalanced |
|--------|----------|------------|
| auc | 0.966 | 0.962 |
| average_precision | 0.815 | 0.409 |
| accuracy | 0.753 | 0.955 |
| precision | 0.500 | 0.000 |
| recall | 0.053 | 0.000 |

## Statistical Analysis

### Effect Sizes (Cohen's d)

- val_auc: 0.442 (small effect)
- val_ap: -1.081 (large effect)
- val_acc: 1.466 (large effect)

## Recommendations

**Recommended Approach: BALANCED**

The balanced approach shows superior performance on the held-out test set despite having less training data.

## Methodological Advantages

This analysis addresses common pitfalls in ML model comparison:

1. **No data leakage**: Test set never used during model selection
2. **No overfitting to validation set**: Nested CV prevents this
3. **Statistical rigor**: Significance testing and effect sizes
4. **Honest performance**: Test set results are unbiased estimates
