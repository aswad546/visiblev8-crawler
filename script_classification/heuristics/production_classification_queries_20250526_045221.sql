
-- PRODUCTION CLASSIFICATION RESULTS ANALYSIS QUERIES
-- Generated: 2025-05-26 04:52:34.155757
-- Table: production_classification_results_20250526_045221

-- 1. HIGH CONFIDENCE BEHAVIORAL BIOMETRIC SCRIPTS (most likely to be malicious)
SELECT 
    script_id, 
    script_url, 
    behavioral_probability,
    max_api_aggregation_score,
    behavioral_api_agg_count,
    fingerprinting_source_api_count
FROM production_classification_results_20250526_045221
WHERE confidence_level = 'High Confidence Behavioral'
ORDER BY behavioral_probability DESC
LIMIT 20;

-- 2. SUMMARY BY CONFIDENCE LEVEL
SELECT 
    confidence_level,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM production_classification_results_20250526_045221), 2) as percentage,
    ROUND(AVG(behavioral_probability), 3) as avg_probability,
    ROUND(AVG(max_api_aggregation_score), 1) as avg_aggregation_score
FROM production_classification_results_20250526_045221
GROUP BY confidence_level
ORDER BY count DESC;

-- 3. TOP DOMAINS BY BEHAVIORAL BIOMETRIC PROBABILITY
SELECT 
    REGEXP_REPLACE(script_url, '^https?://([^/]+).*', '\1') as domain,
    COUNT(*) as script_count,
    ROUND(AVG(behavioral_probability), 3) as avg_probability,
    MAX(behavioral_probability) as max_probability,
    SUM(CASE WHEN confidence_level = 'High Confidence Behavioral' THEN 1 ELSE 0 END) as high_conf_behavioral_count
FROM production_classification_results_20250526_045221
WHERE script_url IS NOT NULL AND script_url != 'Unknown'
GROUP BY REGEXP_REPLACE(script_url, '^https?://([^/]+).*', '\1')
HAVING COUNT(*) >= 2
ORDER BY avg_probability DESC
LIMIT 25;

-- 4. UNCERTAIN CASES THAT NEED MANUAL REVIEW
SELECT 
    script_id, 
    script_url, 
    behavioral_probability,
    max_api_aggregation_score,
    behavioral_api_agg_count
FROM production_classification_results_20250526_045221
WHERE confidence_level = 'Uncertain'
    AND behavioral_probability BETWEEN 0.3 AND 0.7
ORDER BY behavioral_probability DESC
LIMIT 30;

-- 5. POTENTIAL FALSE NEGATIVES (low scores but some behavioral signals)
SELECT 
    script_id, 
    script_url, 
    behavioral_probability,
    behavioral_api_agg_count,
    fingerprinting_source_api_count,
    behavioral_event_diversity
FROM production_classification_results_20250526_045221
WHERE confidence_level = 'High Confidence Normal'
    AND (behavioral_api_agg_count > 5 OR fingerprinting_source_api_count > 15)
ORDER BY behavioral_api_agg_count DESC
LIMIT 20;

-- 6. GET ORIGINAL SCRIPT DETAILS FOR HIGH CONFIDENCE BEHAVIORAL
SELECT 
    r.script_id,
    r.script_url,
    r.behavioral_probability,
    o.behavioral_source_apis,
    o.fingerprinting_source_apis,
    o.apis_going_to_sink
FROM production_classification_results_20250526_045221 r
JOIN multicore_static_info_known_companies o ON r.script_id = o.script_id
WHERE r.confidence_level = 'High Confidence Behavioral'
ORDER BY r.behavioral_probability DESC
LIMIT 10;
