{{
  config(
    materialized='view',
    tags=['staging', 'llm']
  )
}}

with source as (
    select * from read_parquet('/data/chimera-lake/raw/llm_interactions/**/*.parquet')
),

deduplicated as (
    select
        interaction_id,
        session_id,
        tenant_id,
        provider,
        model,
        prompt,
        prompt_hash,
        response,
        system_instruction,
        config,
        tokens_prompt,
        tokens_completion,
        tokens_total,
        latency_ms,
        status,
        error_message,
        created_at,
        ingested_at,
        dt,
        hour,
        -- Add deduplication window function
        row_number() over (
            partition by interaction_id
            order by ingested_at desc
        ) as row_num
    from source
    where created_at >= '{{ var("start_date") }}'
),

final as (
    select
        interaction_id,
        session_id,
        coalesce(tenant_id, 'default') as tenant_id,
        lower(provider) as provider,
        model,
        prompt,
        prompt_hash,
        response,
        system_instruction,
        config,
        coalesce(tokens_prompt, 0) as tokens_prompt,
        coalesce(tokens_completion, 0) as tokens_completion,
        coalesce(tokens_total, 0) as tokens_total,
        latency_ms,
        status,
        error_message,
        cast(created_at as timestamp) as created_at,
        cast(ingested_at as timestamp) as ingested_at,
        dt,
        hour
    from deduplicated
    where row_num = 1  -- Keep only most recent version
)

select * from final
