{{ config(
    materialized='incremental',
    unique_key='usage_key',
    tags=['marts', 'facts', 'usage'],
    on_schema_change='append_new_columns'
) }}

/*
 * Fact Table: Daily Usage Metrics
 *
 * Purpose: Daily aggregated usage metrics for billing and analytics
 *
 * Grain: One row per tenant per provider per day
 */

with source as (
    select * from {{ ref('int_llm_interactions_enriched') }}
    {% if is_incremental() %}
    where created_at > (select max(date) from {{ this }})
    {% endif %}
),

aggregated as (
    select
        -- Keys
        {{ dbt_utils.generate_surrogate_key(['tenant_id', 'provider', 'date']) }} as usage_key,
        tenant_id,
        provider,
        date,

        -- Request volume
        count(*) as total_requests,
        count(case when status = 'success' then 1 end) as successful_requests,
        count(case when status = 'error' then 1 end) as error_requests,
        count(case when status = 'timeout' then 1 end) as timeout_requests,

        -- Token metrics
        sum(tokens_total) as total_tokens,
        sum(tokens_prompt) as total_tokens_prompt,
        sum(tokens_completion) as total_tokens_completion,
        avg(tokens_total) as avg_tokens_per_request,

        -- Token buckets distribution
        sum(case when token_bucket = 'micro' then 1 else 0 end) as micro_requests,
        sum(case when token_bucket = 'small' then 1 else 0 end) as small_requests,
        sum(case when token_bucket = 'medium' then 1 else 0 end) as medium_requests,
        sum(case when token_bucket = 'large' then 1 else 0 end) as large_requests,
        sum(case when token_bucket = 'xlarge' then 1 else 0 end) as xlarge_requests,

        -- Latency metrics
        avg(latency_ms) as avg_latency_ms,
        percentile_cont(0.50) within group (order by latency_ms) as p50_latency_ms,
        percentile_cont(0.95) within group (order by latency_ms) as p95_latency_ms,
        percentile_cont(0.99) within group (order by latency_ms) as p99_latency_ms,

        -- Latency bucket distribution
        sum(case when latency_category = 'fast' then 1 else 0 end) as fast_requests,
        sum(case when latency_category = 'normal' then 1 else 0 end) as normal_latency_requests,
        sum(case when latency_category = 'slow' then 1 else 0 end) as slow_requests,
        sum(case when latency_category = 'timeout_risk' then 1 else 0 end) as timeout_risk_requests,

        -- Model diversity
        count(distinct model) as unique_models,
        count(distinct session_id) as unique_sessions,

        -- Cost estimation (rough calculation)
        -- Note: Actual costs depend on provider pricing
        sum(tokens_prompt) * 0.0005 + sum(tokens_completion) * 0.0015 as estimated_cost_usd,

        -- Timestamps
        min(created_at) as first_request_at,
        max(created_at) as last_request_at,
        current_timestamp as updated_at

    from source
    group by 1, 2, 3, 4
)

select * from aggregated
