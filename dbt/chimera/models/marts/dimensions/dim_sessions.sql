{{ config(
    materialized='table',
    tags=['marts', 'dimensions', 'session']
) }}

/*
 * Dimension Table: Sessions
 *
 * Purpose: Session-level dimension for user interaction analysis
 *
 * Contains:
 * - Session metadata
 * - Aggregate metrics per session
 * - Session characteristics
 */

with session_interactions as (
    -- Get all interactions grouped by session
    select
        session_id,
        tenant_id,
        provider,
        count(*) as total_requests,
        count(distinct model) as unique_models,
        sum(tokens_total) as total_tokens,
        avg(latency_ms) as avg_latency_ms,
        percentile_cont(0.95) within group (order by latency_ms) as p95_latency_ms,
        sum(case when status = 'success' then 1 else 0 end) as successful_requests,
        sum(case when status != 'success' then 1 else 0 end) as failed_requests,
        min(created_at) as first_request_at,
        max(created_at) as last_request_at,
        current_timestamp as calculated_at
    from {{ ref('stg_llm_interactions') }}
    where session_id is not null
    group by 1, 2, 3
),

session_duration as (
    -- Calculate session duration in minutes
    select
        *,
        extract(epoch from (last_request_at - first_request_at)) / 60 as session_duration_minutes,
        date_trunc('day', first_request_at) as session_date
    from session_interactions
),

enriched as (
    select
        -- Surrogate key
        {{ dbt_utils.generate_surrogate_key(['session_id']) }} as session_key,

        -- Natural key
        session_id,

        -- Tenant
        tenant_id,

        -- Session characteristics
        case
            when total_requests = 1 then 'single_request'
            when total_requests <= 5 then 'short'
            when total_requests <= 20 then 'medium'
            else 'long'
        end as session_type,

        case
            when unique_models = 1 then 'single_model'
            when unique_models <= 3 then 'few_models'
            else 'multi_model'
        end as model_diversity,

        -- Duration bucketing
        case
            when session_duration_minutes < 1 then 'instant'
            when session_duration_minutes < 5 then 'brief'
            when session_duration_minutes < 30 then 'normal'
            when session_duration_minutes < 120 then 'extended'
            else 'marathon'
        end as duration_bucket,

        -- Metrics
        total_requests,
        unique_models,
        total_tokens,
        avg_latency_ms,
        p95_latency_ms,
        successful_requests,
        failed_requests,

        -- Success rate
        (failed_requests::float / nullif(total_requests, 0)) as failure_rate,

        -- Timestamps
        first_request_at,
        last_request_at,
        session_duration_minutes,
        session_date,
        calculated_at as updated_at

    from session_duration
)

select * from enriched
