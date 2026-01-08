{{ config(
    materialized='incremental',
    unique_key='experiment_key',
    tags=['marts', 'aggregations', 'jailbreak'],
    on_schema_change='append_new_columns'
) }}

/*
 * Aggregation: Jailbreak Research Analytics
 *
 * Purpose: Track jailbreak experiment success rates and metrics
 *
 * Grain: One row per framework per attack method per day
 */

with source as (
    select * from {{ ref('stg_jailbreak_experiments') }}
    {% if is_incremental() %}
    where created_at > (select max(date) from {{ this }})
    {% endif %}
),

aggregated as (
    select
        -- Keys
        {{ dbt_utils.generate_surrogate_key(['framework', 'attack_method', 'date']) }} as experiment_key,
        framework,
        attack_method,
        date,

        -- Experiment volume
        count(*) as total_experiments,

        -- Success metrics
        sum(case when success then 1 else 0 end) as successful_experiments,
        sum(case when not success then 1 else 0 end) as failed_experiments,
        (sum(case when success then 1 else 0 end)::float / count(*)) as success_rate,

        -- Judge score metrics (if applicable)
        avg(judge_score) as avg_judge_score,
        max(judge_score) as max_judge_score,
        percentile_cont(0.50) within group (order by judge_score) as median_judge_score,

        -- Iteration metrics
        avg(iterations) as avg_iterations,
        min(iterations) as min_iterations,
        max(iterations) as max_iterations,

        -- Iteration buckets
        sum(case when iterations = 0 then 1 else 0 end) as zero_iteration_experiments,
        sum(case when iterations between 1 and 5 then 1 else 0 end) as low_iteration_experiments,
        sum(case when iterations between 6 and 20 then 1 else 0 end) as medium_iteration_experiments,
        sum(case when iterations > 20 then 1 else 0 end) as high_iteration_experiments,

        -- Framework effectiveness ranking
        row_number() over (
            partition by date
            order by (sum(case when success then 1 else 0 end)::float / count(*)) desc
        ) as framework_rank_daily,

        -- Timestamps
        min(created_at) as first_experiment_at,
        max(created_at) as last_experiment_at,
        current_timestamp as updated_at

    from source
    group by 1, 2, 3, 4
)

select * from aggregated
