{{ config(
    materialized='view',
    tags=['staging']
) }}

with source as (
    -- Read from Delta Lake table
    select * from {{ source('delta_lake', 'transformations') }}
    where ingested_at >= '{{ var("start_date") }}'
),

deduplicated as (
    select *,
        row_number() over (
            partition by transformation_id
            order by ingested_at desc
        ) as row_num
    from source
)

select *
from deduplicated
where row_num = 1
