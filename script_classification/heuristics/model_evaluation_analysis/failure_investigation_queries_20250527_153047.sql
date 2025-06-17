-- MODEL EVALUATION: FAILURE INVESTIGATION QUERIES
-- Generated: 2025-05-27 15:31:07.006903
-- Model: Random Forest (IMBALANCED Dataset - RECOMMENDED)

-- FALSE POSITIVE INVESTIGATION
-- Normal scripts incorrectly flagged as behavioral biometric

-- High-confidence false positives (detailed analysis)
SELECT 
    script_id,
    script_url,
    max_api_aggregation_score,
    behavioral_api_agg_count,
    fingerprinting_source_api_count,
    total_fp_api_accesses,
    behavioral_source_apis,
    fingerprinting_source_apis,
    apis_going_to_sink,
    label
FROM multicore_static_info_known_companies
WHERE script_id IN (7392094, 7396886, 7395579, 7392038, 7395959, 7393356, 7400941, 7411592, 7412821, 7395905)
ORDER BY fingerprinting_source_api_count DESC;

-- Domain analysis for false positives
SELECT 
    REGEXP_REPLACE(script_url, '^https?://([^/]+).*', '\1') as domain,
    COUNT(*) as fp_count,
    AVG(fingerprinting_source_api_count) as avg_fp_apis
FROM multicore_static_info_known_companies
WHERE script_id IN (7392094, 7396886, 7395579, 7392038, 7395959, 7393356, 7400941, 7411592, 7412821, 7395905)
    AND script_url IS NOT NULL
GROUP BY REGEXP_REPLACE(script_url, '^https?://([^/]+).*', '\1')
ORDER BY fp_count DESC;

-- GENERAL INVESTIGATION QUERIES

-- Scripts with similar patterns to false positives
SELECT script_id, script_url, total_fp_api_accesses, max_api_aggregation_score
FROM multicore_static_info_known_companies
WHERE label = 0
  AND total_fp_api_accesses > 85
  AND max_api_aggregation_score > 19
ORDER BY total_fp_api_accesses DESC
LIMIT 20;

-- Scripts with weak biometric signals (potential mislabels)
SELECT script_id, script_url, behavioral_api_agg_count, max_api_aggregation_score
FROM multicore_static_info_known_companies
WHERE label = 1
  AND behavioral_api_agg_count < 5
  AND max_api_aggregation_score < 10
ORDER BY behavioral_api_agg_count ASC
LIMIT 20;

