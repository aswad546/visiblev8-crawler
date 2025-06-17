# Behavioral Biometric Detection Analysis Report

**Generated:** 2025-05-26 02:37:12.667679
**Model Type:** Random Forest
**Analysis ID:** 20250526_023608

## Executive Summary

- **Overall Accuracy:** 96.3%
- **False Positive Rate:** 1.5%
- **False Negative Rate:** 2.2%
- **Model Robustness:** Excellent

## Key Findings

### Most Important Features
- **fingerprinting_source_api_count:** 0.187
- **total_fp_api_accesses:** 0.154
- **max_api_aggregation_score:** 0.148
- **unique_fp_apis:** 0.134
- **mouse_event_count:** 0.070

### Detection Patterns
- Behavioral biometric scripts show distinctive aggregation patterns
- Graph construction success is a strong differentiator
- API diversity and intensity are key indicators

## Recommendations

### For Production Deployment
1. Use the trained Random Forest model with 95%+ confidence threshold
2. Implement secondary validation for edge cases
3. Monitor false positive rates in production

### For Model Improvement
1. Investigate false positives to refine feature engineering
2. Analyze label=-1 samples for potential new positives
3. Consider temporal features for next iteration

## Technical Implementation

### Model Configuration
- **bootstrap:** True
- **ccp_alpha:** 0.0
- **class_weight:** balanced
- **criterion:** gini
- **max_depth:** 10
- **max_features:** log2
- **max_leaf_nodes:** None
- **max_samples:** None
- **min_impurity_decrease:** 0.0
- **min_samples_leaf:** 3
- **min_samples_split:** 2
- **min_weight_fraction_leaf:** 0.0
- **monotonic_cst:** None
- **n_estimators:** 300
- **n_jobs:** -1
- **oob_score:** False
- **random_state:** 42
- **verbose:** 0
- **warm_start:** False

### Feature Engineering
- **Total Features:** 21
- **Feature Categories:** Aggregation, Volume, Flow, Diversity
- **Data Quality:** Graph construction analysis performed

## Generated Files

### Visualizations
- `analysis_20250526_023608_feature_correlations.png`
- `analysis_20250526_023608_cv_analysis_random_forest.png`
- `analysis_20250526_023608_feature_distributions.png`
- `analysis_20250526_023608_cv_analysis_decision_tree.png`
- `analysis_20250526_023608_misclassification_patterns.png`
- `analysis_20250526_023608_feature_importance_analysis.png`

### Data Files
- `best_model_20250526_023608.pkl`
- `investigation_queries_20250526_023608.sql`
