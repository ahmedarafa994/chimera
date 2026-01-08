{{
  config(
    materialized='incremental',
    unique_key='agg_key',
    tags=['marts', 'aggregation', 'provider'],
    on_schema_change='append_new_columns'
  )
}}

with llm_interactions as (
    select * from {{ ref('stg_llm_interactions') }}
    {% if is_incremental() %}
    where created_at > (select max(hour_timestamp) from {{ this }})
    {% endif %}
),

hourly_aggregated as (
    select
        provider,
        model,
        date_trunc('hour', created_at) as hour_timestamp,
        dt,

        -- Volume metrics
        count(*) as total_requests,
        count(case when status = 'success' then 1 end) as successful_requests,
        count(case when status = 'error' then 1 end) as failed_requests,
        count(case when status = 'timeout' then 1 end) as timeout_requests,

        -- Token metrics
        sum(tokens_total) as total_tokens,
        sum(tokens_prompt) as total_tokens_prompt,
        sum(tokens_completion) as total_tokens_completion,
        avg(tokens_total) as avg_tokens_per_request,

        -- Latency metrics
        avg(latency_ms) as avg_latency_ms,
        percentile_cont(0.5) within group (order by latency_ms) as p50_latency_ms,
        percentile_cont(0.95) within group (order by latency_ms) as p95_latency_ms,
        percentile_cont(0.99) within group (order by latency_ms) as p99_latency_ms,
        min(latency_ms) as min_latency_ms,
        max(latency_ms) as max_latency_ms,

        -- Error rate
        (count(case when status != 'success' then 1 end)::float / count(*)) as error_rate,

        -- Unique sessions
        count(distinct session_id) as unique_sessions,
        count(distinct tenant_id) as unique_tenants

    from llm_interactions
    group by 1, 2, 3, 4
),

final as (
    select
        {{ dbt_utils.generate_surrogate_key(['provider', 'model', 'hour_timestamp']) }} as agg_key,
        *,
        current_timestamp as updated_at
    from hourly_aggregated
)

select * from final
