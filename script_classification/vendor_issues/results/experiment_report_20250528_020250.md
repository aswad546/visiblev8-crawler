# Vendor-Aware Model Comparison Experiment Report

**Generated:** 2025-05-28 02:02:50

## Executive Summary

- **Best Approach:** Balanced Small
- **Best F1 Score:** 1.000
- **Test AUC:** 1.000
- **Optimal Threshold:** 0.250

## Key Findings

1. **Threshold Optimization Critical**: Default 0.5 threshold severely limits recall
2. **Vendor Composition Matters**: Removing dominant vendors can improve generalization
3. **Vendor-Aware Evaluation**: Essential for realistic performance estimates

## Full Results

      Approach  Val_AUC  Test_AUC  Test_AP  Default_Precision  Default_Recall  Default_F1  Optimal_Precision  Optimal_Recall  Optimal_F1  Optimal_Threshold
Balanced Small      1.0     1.000    1.000                0.0             0.0         0.0              1.000            1.00       1.000               0.25
  Balanced All      1.0     0.929    0.984                0.0             0.0         0.0              0.976            1.00       0.988               0.03
Imbalanced All      1.0     0.950    0.743                0.0             0.0         0.0              0.810            0.84       0.824               0.04

## Methodology

- Vendor-grouped train/test splits prevent data leakage
- Models tested on completely unseen vendors
- Threshold optimization using F1 score
- Three dataset variants tested
