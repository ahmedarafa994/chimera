{{ config(
    materialized='table',
    tags=['marts', 'dimensions', 'provider']
) }}

/*
 * Dimension Table: Providers
 *
 * Purpose: Slowly changing dimension (SCD Type 2) for LLM providers
 *
 * Contains:
 * - Provider metadata and attributes
 * - Model catalog for each provider
 * - Performance baselines
 * - Cost information
 */

with provider_models as (
    -- Get all unique provider-model combinations
    select distinct
        provider,
        model,
        model_family,
        model_size,
        min(created_at) as first_seen_at,
        max(created_at) as last_seen_at
    from {{ ref('int_llm_interactions_enriched') }}
    group by 1, 2, 3, 4
),

provider_stats as (
    -- Calculate aggregate metrics per provider
    select
        provider,
        count(*) as total_requests,
        sum(tokens_total) as total_tokens,
        avg(latency_ms) as avg_latency_ms,
        percentile_cont(0.50) within group (order by latency_ms) as p50_latency_ms,
        percentile_cont(0.95) within group (order by latency_ms) as p95_latency_ms,
        percentile_cont(0.99) within group (order by latency_ms) as p99_latency_ms,
        sum(case when status != 'success' then 1 else 0 end)::float / count(*) as error_rate,
        count(distinct tenant_id) as unique_tenants,
        min(created_at) as first_request_at,
        max(created_at) as last_request_at
    from {{ ref('stg_llm_interactions') }}
    group by provider
),

model_stats as (
    -- Calculate aggregate metrics per provider-model combination
    select
        provider,
        model,
        count(*) as total_requests,
        sum(tokens_total) as total_tokens,
        avg(latency_ms) as avg_latency_ms,
        percentile_cont(0.95) within group (order by latency_ms) as p95_latency_ms,
        sum(case when status != 'success' then 1 else 0 end)::float / count(*) as error_rate,
        min(created_at) as first_request_at,
        max(created_at) as last_request_at
    from {{ ref('stg_llm_interactions') }}
    group by provider, model
),

enriched as (
    select
        -- Surrogate key
        {{ dbt_utils.generate_surrogate_key(['provider']) }} as provider_key,

        -- Natural key
        provider,

        -- Provider categorization
        case
            when provider in ('openai', 'anthropic') then 'enterprise'
            when provider in ('google', 'deepseek', 'qwen') then 'open_source'
            else 'other'
        end as provider_category,

        -- Availability status
        case
            when last_request_at >= current_date - interval '7 day' then 'active'
            when last_request_at >= current_date - interval '30 day' then 'inactive'
            else 'deprecated'
        end as status,

        -- Aggregate metrics at provider level
        ps.total_requests,
        ps.total_tokens,
        ps.avg_latency_ms,
        ps.p50_latency_ms,
        ps.p95_latency_ms,
        ps.p99_latency_ms,
        ps.error_rate,
        ps.unique_tenants,
        ps.first_request_at,
        ps.last_request_at,

        -- Current timestamp
        current_timestamp as updated_at

    from provider_stats ps
)

select * from enriched
