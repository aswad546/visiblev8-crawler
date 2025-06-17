-- Behavioral Biometric Detection: Misclassification Investigation Queries
-- Generated: 2025-05-26 02:37:12.658876
-- Model: Random Forest

-- FALSE POSITIVES: Normal scripts incorrectly flagged as behavioral biometric
-- These may reveal weaknesses in your detection logic

-- Query to examine false positive scripts in detail
SELECT 
    script_id,
    script_url,
    max_api_aggregation_score,
    behavioral_api_agg_count,
    fingerprinting_source_api_count,
    behavioral_source_apis,
    apis_going_to_sink,
    label
FROM multicore_static_info_known_companies
WHERE script_id IN (7412386, 7398439)
ORDER BY max_api_aggregation_score DESC;

-- High-confidence false positives (most concerning)
SELECT script_id, script_url, code
FROM multicore_static_info_known_companies
WHERE script_id IN (7412386, 7398439);

-- FALSE NEGATIVES: Behavioral biometric scripts that were missed
-- These reveal gaps in your detection capabilities

-- Query to examine false negative scripts in detail
SELECT 
    script_id,
    script_url,
    max_api_aggregation_score,
    behavioral_api_agg_count,
    fingerprinting_source_api_count,
    behavioral_source_apis,
    apis_going_to_sink,
    label
FROM multicore_static_info_known_companies
WHERE script_id IN (7397523, 7413095, 7397312)
ORDER BY max_api_aggregation_score ASC;

-- Near-miss false negatives (close to detection threshold)
SELECT script_id, script_url, code
FROM multicore_static_info_known_companies
WHERE script_id IN (7397523, 7413095, 7397312);

-- GENERAL INVESTIGATION QUERIES

-- Find scripts with similar patterns to false positives
SELECT script_id, script_url, max_api_aggregation_score, behavioral_api_agg_count
FROM multicore_static_info_known_companies
WHERE label = 0 
  AND max_api_aggregation_score > 10
  AND behavioral_api_agg_count > 5
ORDER BY max_api_aggregation_score DESC
LIMIT 20;

-- Find potentially mislabeled scripts (label=-1 with high biometric signals)
SELECT script_id, script_url, max_api_aggregation_score, behavioral_api_agg_count
FROM multicore_static_info_known_companies
WHERE label = -1 
  AND max_api_aggregation_score > 15
  AND fingerprinting_source_api_count > 10
  AND graph_construction_failure = false
ORDER BY max_api_aggregation_score DESC
LIMIT 50;

