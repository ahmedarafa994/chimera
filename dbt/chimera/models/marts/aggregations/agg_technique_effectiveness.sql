{{ config(
    materialized='incremental',
    unique_key='technique_key',
    tags=['marts', 'aggregations', 'technique'],
    on_schema_change='append_new_columns'
) }}

/*
 * Aggregation: Transformation Technique Effectiveness
 *
 * Purpose: Analyze the performance and success rates of transformation techniques
 *
 * Grain: One row per technique suite per day
 */

with source as (
    select * from {{ ref('int_transformations_enriched') }}
    {% if is_incremental() %}
    where created_at > (select max(date) from {{ this }})
    {% endif %}
),

aggregated as (
    select
        -- Keys
        {{ dbt_utils.generate_surrogate_key(['technique_suite', 'date']) }} as technique_key,
        technique_suite,
        technique_category,
        date,

        -- Usage volume
        count(*) as total_transformations,

        -- Success metrics
        sum(case when success then 1 else 0 end) as successful_transformations,
        sum(case when not success then 1 else 0 end) as failed_transformations,
        (sum(case when success then 1 else 0 end)::float / count(*)) as success_rate,

        -- Performance metrics
        avg(transformation_time_ms) as avg_transformation_time_ms,
        percentile_cont(0.50) within group (order by transformation_time_ms) as p50_transformation_time_ms,
        percentile_cont(0.95) within group (order by transformation_time_ms) as p95_transformation_time_ms,

        -- Potency distribution
        avg(potency_level) as avg_potency_level,
        sum(case when potency_level <= 2 then 1 else 0 end) as low_potency_count,
        sum(case when potency_level between 3 and 5 then 1 else 0 end) as medium_potency_count,
        sum(case when potency_level >= 6 then 1 else 0 end) as high_potency_count,

        -- Expansion metrics
        avg(expansion_ratio) as avg_expansion_ratio,
        avg(original_prompt_length) as avg_input_length,
        avg(transformed_prompt_length) as avg_output_length,

        -- Iteration stats (from metadata)
        avg(iterations) as avg_iterations,
        max(iterations) as max_iterations,

        -- Associated interaction count
        count(distinct interaction_id) as unique_interactions,

        -- Timestamps
        min(created_at) as first_transformation_at,
        max(created_at) as last_transformation_at,
        current_timestamp as updated_at

    from source
    group by 1, 2, 3, 4
)

select * from aggregated
