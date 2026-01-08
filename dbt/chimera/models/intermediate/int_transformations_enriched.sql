{{ config(
    materialized='view',
    tags=['intermediate', 'enrichment', 'transformations']
) }}

/*
 * Intermediate Model: Enriched Transformation Events
 *
 * Purpose: Add business logic and analysis fields to transformation events.
 *
 * Transformations:
 * - Technique suite categorization
 * - Success rate analysis
 * - Performance metrics
 * - Potency scoring
 */

with source as (
    select * from {{ ref('stg_transformations') }}
    where created_at >= '{{ var("start_date") }}'
),

enriched as (
    select
        -- Primary keys
        transformation_id,
        interaction_id,
        technique_suite,
        technique_name,

        -- Technique categorization
        case
            when technique_suite in ('simple', 'basic') then 'basic'
            when technique_suite in ('advanced', 'expert') then 'advanced'
            when technique_suite in ('cognitive', 'hypothetical_scenario') then 'cognitive'
            when technique_suite in ('obfuscation', 'advanced_obfuscation', 'typoglycemia') then 'obfuscation'
            when technique_suite in ('persona', 'hierarchical_persona', 'dan_persona') then 'persona'
            when technique_suite in ('context', 'contextual_inception', 'nested_context') then 'context'
            when technique_suite in ('logic', 'logical_inference', 'conditional_logic') then 'logic'
            when technique_suite in ('multimodal', 'visual_context') then 'multimodal'
            when technique_suite in ('agentic', 'multi_agent') then 'agentic'
            when technique_suite in ('payload', 'instruction_fragmentation') then 'payload'
            else 'other'
        end as technique_category,

        -- Potency level (from metadata if available)
        coalesce(
            cast(json_extract(metadata, '$.potency') as integer),
            case
                when technique_suite in ('simple', 'basic') then 1
                when technique_suite in ('advanced', 'expert') then 3
                when technique_suite in ('obfuscation', 'persona') then 5
                when technique_suite in ('agentic', 'payload') then 7
                when technique_suite in ('cognitive', 'context') then 6
                else 4
            end
        ) as potency_level,

        -- Content
        original_prompt,
        transformed_prompt,

        -- Length metrics
        length(original_prompt) as original_prompt_length,
        length(transformed_prompt) as transformed_prompt_length,

        -- Expansion ratio (how much the prompt grew)
        case
            when length(original_prompt) > 0
            then length(transformed_prompt)::float / length(original_prompt)
            else null
        end as expansion_ratio,

        -- Performance
        transformation_time_ms,

        -- Performance categorization
        case
            when transformation_time_ms < 50 then 'instant'
            when transformation_time_ms < 200 then 'fast'
            when transformation_time_ms < 1000 then 'normal'
            else 'slow'
        end as performance_category,

        -- Success status
        success,

        -- Metadata (parsed as JSON)
        metadata,

        -- Extract iteration count if available
        coalesce(
            cast(json_extract(metadata, '$.iterations') as integer),
            1
        ) as iterations,

        -- Timestamps
        created_at,
        ingested_at,
        dt,
        hour,
        date(created_at) as date,
        date_trunc('hour', created_at) as hour_timestamp

    from source
)

select * from enriched
