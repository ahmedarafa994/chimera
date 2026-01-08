{{ config(
    materialized='view',
    tags=['intermediate', 'enrichment']
) }}

/*
 * Intermediate Model: Enriched LLM Interactions
 *
 * Purpose: Add business logic, categorization, and derived fields
 *          to raw LLM interaction data for downstream analytics.
 *
 * Transformations:
 * - Provider categorization (enterprise vs open source)
 * - Model family grouping (gemini, gpt, claude, etc.)
 * - Token efficiency metrics
 * - Latency categorization
 * - Request size buckets
 * - Error categorization
 */

with source as (
    select * from {{ ref('stg_llm_interactions') }}
    where created_at >= '{{ var("start_date") }}'
),

enriched as (
    select
        -- Primary keys and identifiers
        interaction_id,
        session_id,
        tenant_id,
        provider,
        model,

        -- Provider categorization
        case
            when provider in ('openai', 'anthropic') then 'enterprise'
            when provider in ('google', 'deepseek', 'qwen') then 'open_source'
            else provider
        end as provider_category,

        -- Model family extraction
        case
            when model like '%gemini%' then 'gemini'
            when model like '%gpt%' or model like '%chatgpt%' then 'gpt'
            when model like '%claude%' or model like '%anthropic%' then 'claude'
            when model like '%deepseek%' then 'deepseek'
            when model like '%qwen%' then 'qwen'
            when model like '%mock%' then 'mock'
            else 'other'
        end as model_family,

        -- Model version/size classification
        case
            when model like '%flash%' or model like '%haiku%' or model like '%lite%' then 'small'
            when model like '%pro%' or model like '%sonnet%' then 'medium'
            when model like '%ultra%' or model like '%opus%' then 'large'
            else 'unknown'
        end as model_size,

        -- Content metrics
        prompt,
        response,
        system_instruction,
        config,

        -- Token metrics
        tokens_prompt,
        tokens_completion,
        tokens_total,

        -- Derived: Token efficiency (response per input token)
        case
            when tokens_prompt > 0
            then tokens_completion::float / nullif(tokens_prompt, 0)
            else null
        end as token_efficiency_ratio,

        -- Derived: Token cost bucket (for cost analysis)
        case
            when tokens_total < 500 then 'micro'
            when tokens_total < 2000 then 'small'
            when tokens_total < 8000 then 'medium'
            when tokens_total < 32000 then 'large'
            else 'xlarge'
        end as token_bucket,

        -- Performance metrics
        latency_ms,

        -- Latency categorization
        case
            when latency_ms < 500 then 'fast'
            when latency_ms < 2000 then 'normal'
            when latency_ms < 10000 then 'slow'
            else 'timeout_risk'
        end as latency_category,

        -- Derived: Tokens per second (throughput metric)
        case
            when latency_ms > 0
            then (tokens_total::float / (latency_ms::float / 1000.0))
            else null
        end as tokens_per_second,

        -- Status categorization
        status,

        -- Error categorization
        case
            when status = 'success' then 'no_error'
            when error_message like '%timeout%' then 'timeout'
            when error_message like '%rate limit%' then 'rate_limit'
            when error_message like '%quota%' then 'quota_exceeded'
            when error_message like '%authentication%' or error_message like '%auth%' then 'auth_error'
            when error_message like '%permission%' or error_message like '%access%' then 'permission_error'
            else 'other_error'
        end as error_category,

        error_message,

        -- Time-based fields
        created_at,
        ingested_at,
        dt,
        hour,

        -- Derived time fields
        date(created_at) as date,
        date_trunc('hour', created_at) as hour_timestamp,
        date_trunc('day', created_at) as day_timestamp,
        date_trunc('week', created_at) as week_timestamp,
        date_trunc('month', created_at) as month_timestamp,

        -- Derived: Day of week
        extract(dow from created_at) as day_of_week,

        -- Derived: Hour of day (for time-of-day analysis)
        extract(hour from created_at) as hour_of_day,

        -- Time period classification
        case
            when extract(hour from created_at) >= 6 and extract(hour from created_at) < 12 then 'morning'
            when extract(hour from created_at) >= 12 and extract(hour from created_at) < 18 then 'afternoon'
            when extract(hour from created_at) >= 18 and extract(hour from created_at) < 22 then 'evening'
            else 'night'
        end as time_period,

        -- Weekday vs weekend
        case
            when extract(dow from created_at) in (0, 6) then 'weekend'
            else 'weekday'
        end as day_type

    from source
)

select * from enriched
