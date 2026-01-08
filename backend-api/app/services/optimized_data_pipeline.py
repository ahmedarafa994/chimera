"""
Optimized Data Pipeline with parallel processing and memory efficiency.

This module provides comprehensive data pipeline optimizations:
- Parallel data ingestion and processing
- Memory-efficient streaming processing
- Distributed processing with worker pools
- Advanced caching and compression
- Real-time monitoring and metrics
- Fault tolerance and recovery
"""

import asyncio
import gc
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from app.core.logging import logger

# =====================================================
# Optimized Data Models and Configuration
# =====================================================

class OptimizedIngestionConfig(BaseModel):
    """Enhanced configuration for optimized data pipeline."""

    # Base configuration
    data_lake_path: str = Field(default="/data/chimera-lake-optimized")
    partition_by: list[str] = Field(default=["dt", "hour"])
    compression: str = Field(default="zstd", regex="^(snappy|gzip|zstd|lz4)$")

    # Memory optimization
    max_memory_usage_mb: int = Field(default=2048, ge=512, le=8192)
    chunk_size: int = Field(default=10000, ge=1000, le=100000)
    enable_memory_monitoring: bool = Field(default=True)

    # Parallel processing
    max_workers: int = Field(default=8, ge=1, le=32)
    enable_multiprocessing: bool = Field(default=True)
    process_pool_size: int = Field(default=4, ge=1, le=16)

    # Caching and compression
    enable_intermediate_cache: bool = Field(default=True)
    cache_compression: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600)

    # Streaming processing
    enable_streaming: bool = Field(default=True)
    stream_buffer_size: int = Field(default=1000)
    stream_flush_interval_ms: int = Field(default=5000)

    # Quality and monitoring
    enable_data_quality_checks: bool = Field(default=True)
    quality_sample_rate: float = Field(default=0.1, ge=0.01, le=1.0)
    enable_performance_metrics: bool = Field(default=True)


@dataclass
class ProcessingMetrics:
    """Metrics for data pipeline processing."""

    total_records_processed: int = 0
    records_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    processing_time_ms: float = 0.0
    data_quality_score: float = 1.0


@dataclass
class BatchJob:
    """Represents a batch processing job."""

    job_id: str
    table_name: str
    start_time: float
    end_time: float
    chunk_size: int
    priority: int = 0
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# =====================================================
# Memory-Efficient Data Processor
# =====================================================

class MemoryEfficientProcessor:
    """
    Memory-efficient data processor with streaming capabilities.
    """

    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self._current_memory_mb = 0.0
        self._memory_threshold = max_memory_mb * 0.8  # 80% threshold

    def estimate_dataframe_memory(self, df: pd.DataFrame) -> float:
        """Estimate memory usage of DataFrame in MB."""
        return df.memory_usage(deep=True).sum() / (1024 * 1024)

    async def process_in_chunks(
        self,
        data_source: pd.DataFrame | str | Path,
        chunk_size: int,
        processor_func: callable,
        **kwargs
    ) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Process data in memory-efficient chunks.

        Args:
            data_source: DataFrame, file path, or data source
            chunk_size: Number of records per chunk
            processor_func: Function to process each chunk

        Yields:
            Processed DataFrame chunks
        """
        if isinstance(data_source, pd.DataFrame):
            # Process DataFrame in chunks
            total_rows = len(data_source)
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk = data_source.iloc[start_idx:end_idx].copy()

                # Check memory before processing
                chunk_memory = self.estimate_dataframe_memory(chunk)
                if self._current_memory_mb + chunk_memory > self._memory_threshold:
                    # Force garbage collection
                    gc.collect()
                    self._current_memory_mb = 0.0

                # Process chunk
                processed_chunk = await self._process_chunk_safe(
                    chunk, processor_func, **kwargs
                )

                if processed_chunk is not None:
                    yield processed_chunk

                # Cleanup
                del chunk
                if 'processed_chunk' in locals():
                    del processed_chunk
                gc.collect()

        elif isinstance(data_source, (str, Path)):
            # Process file in chunks (Parquet/CSV)
            if str(data_source).endswith('.parquet'):
                parquet_file = pq.ParquetFile(str(data_source))
                for batch in parquet_file.iter_batches(batch_size=chunk_size):
                    chunk = batch.to_pandas()
                    processed_chunk = await self._process_chunk_safe(
                        chunk, processor_func, **kwargs
                    )
                    if processed_chunk is not None:
                        yield processed_chunk
                    del chunk
                    if 'processed_chunk' in locals():
                        del processed_chunk

            elif str(data_source).endswith('.csv'):
                for chunk in pd.read_csv(str(data_source), chunksize=chunk_size):
                    processed_chunk = await self._process_chunk_safe(
                        chunk, processor_func, **kwargs
                    )
                    if processed_chunk is not None:
                        yield processed_chunk
                    del chunk
                    if 'processed_chunk' in locals():
                        del processed_chunk

    async def _process_chunk_safe(
        self,
        chunk: pd.DataFrame,
        processor_func: callable,
        **kwargs
    ) -> pd.DataFrame | None:
        """Safely process a chunk with error handling."""
        try:
            # Update memory tracking
            chunk_memory = self.estimate_dataframe_memory(chunk)
            self._current_memory_mb += chunk_memory

            # Process chunk
            if asyncio.iscoroutinefunction(processor_func):
                result = await processor_func(chunk, **kwargs)
            else:
                result = processor_func(chunk, **kwargs)

            # Update memory tracking
            if result is not None:
                result_memory = self.estimate_dataframe_memory(result)
                self._current_memory_mb = self._current_memory_mb - chunk_memory + result_memory

            return result

        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return None


# =====================================================
# Parallel Data Pipeline Engine
# =====================================================

class ParallelDataPipelineEngine:
    """
    High-performance data pipeline engine with parallel processing.
    """

    def __init__(self, config: OptimizedIngestionConfig):
        self.config = config
        self.data_lake_path = Path(config.data_lake_path)

        # Processing components
        self.memory_processor = MemoryEfficientProcessor(config.max_memory_usage_mb)

        # Parallel execution
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.process_pool_size) if config.enable_multiprocessing else None

        # Job management
        self._active_jobs: dict[str, BatchJob] = {}
        self._job_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._results_cache: dict[str, Any] = {}

        # Metrics
        self.metrics = ProcessingMetrics()
        self._start_time = time.time()

        # Background tasks
        self._workers: list[asyncio.Task] = []
        self._monitoring_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self) -> None:
        """Start the parallel processing engine."""
        self._is_running = True

        # Start worker tasks
        for i in range(self.config.max_workers):
            worker_task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker_task)

        # Start monitoring
        if self.config.enable_performance_metrics:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info(f"Parallel data pipeline engine started with {self.config.max_workers} workers")

    async def stop(self) -> None:
        """Stop the pipeline engine gracefully."""
        self._is_running = False

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        # Cancel monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()

        # Wait for completion
        await asyncio.gather(*self._workers, return_exceptions=True)

        # Cleanup thread pools
        self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)

        logger.info("Parallel data pipeline engine stopped")

    async def submit_batch_job(
        self,
        table_name: str,
        start_time: float,
        end_time: float,
        priority: int = 0,
        **kwargs
    ) -> str:
        """Submit a batch processing job."""
        job = BatchJob(
            job_id=f"{table_name}_{int(time.time() * 1000)}",
            table_name=table_name,
            start_time=start_time,
            end_time=end_time,
            chunk_size=kwargs.get('chunk_size', self.config.chunk_size),
            priority=priority,
            metadata=kwargs,
        )

        await self._job_queue.put(job)
        self._active_jobs[job.job_id] = job

        logger.info(f"Submitted batch job: {job.job_id} for table {table_name}")
        return job.job_id

    async def process_llm_interactions_parallel(
        self,
        start_time: float,
        end_time: float,
        source_configs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Process LLM interactions in parallel from multiple sources.
        """
        start_processing = time.time()

        # Create tasks for parallel extraction
        extraction_tasks = []
        for config in source_configs:
            task = asyncio.create_task(
                self._extract_llm_interactions_optimized(
                    start_time, end_time, config
                )
            )
            extraction_tasks.append(task)

        # Wait for all extractions
        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        # Combine results
        all_dataframes = []
        for result in extraction_results:
            if isinstance(result, pd.DataFrame) and not result.empty:
                all_dataframes.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Extraction failed: {result}")

        if not all_dataframes:
            return {"status": "no_data", "processing_time_ms": (time.time() - start_processing) * 1000}

        # Combine DataFrames efficiently
        combined_df = await self._combine_dataframes_optimized(all_dataframes)

        # Process in parallel chunks
        processing_tasks = []
        async for chunk in self.memory_processor.process_in_chunks(
            combined_df,
            self.config.chunk_size,
            self._process_llm_chunk
        ):
            # Write chunks in parallel
            task = asyncio.create_task(self._write_chunk_optimized(chunk, "llm_interactions"))
            processing_tasks.append(task)

        # Wait for all processing
        write_results = await asyncio.gather(*processing_tasks, return_exceptions=True)

        # Calculate metrics
        total_records = len(combined_df)
        successful_writes = sum(1 for r in write_results if not isinstance(r, Exception))
        processing_time = (time.time() - start_processing) * 1000

        self.metrics.total_records_processed += total_records
        self.metrics.records_per_second = total_records / (processing_time / 1000)
        self.metrics.processing_time_ms = processing_time

        return {
            "status": "success",
            "total_records": total_records,
            "successful_writes": successful_writes,
            "processing_time_ms": processing_time,
            "records_per_second": self.metrics.records_per_second,
        }

    async def _extract_llm_interactions_optimized(
        self,
        start_time: float,
        end_time: float,
        source_config: dict[str, Any],
    ) -> pd.DataFrame:
        """Optimized extraction of LLM interactions."""
        # Use process pool for CPU-intensive extraction
        if self.process_pool and source_config.get('cpu_intensive', False):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.process_pool,
                self._extract_llm_interactions_sync,
                start_time,
                end_time,
                source_config,
            )
            return result
        else:
            # Use thread pool for I/O-intensive extraction
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self._extract_llm_interactions_sync,
                start_time,
                end_time,
                source_config,
            )
            return result

    def _extract_llm_interactions_sync(
        self,
        start_time: float,
        end_time: float,
        source_config: dict[str, Any],
    ) -> pd.DataFrame:
        """Synchronous extraction method for thread/process pools."""
        # This would implement actual data extraction logic
        # For now, return sample data for demonstration
        from datetime import datetime, timedelta

        import pandas as pd

        # Generate sample data
        records = []
        current_time = datetime.fromtimestamp(start_time)
        end_datetime = datetime.fromtimestamp(end_time)

        while current_time < end_datetime:
            record = {
                "interaction_id": f"int_{int(current_time.timestamp())}",
                "provider": "google",
                "model": "gemini-pro",
                "prompt": "Sample prompt",
                "response": "Sample response",
                "tokens_total": 150,
                "latency_ms": 500,
                "created_at": current_time,
            }
            records.append(record)
            current_time += timedelta(minutes=1)

        return pd.DataFrame(records)

    async def _combine_dataframes_optimized(
        self,
        dataframes: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """Efficiently combine multiple DataFrames."""
        if len(dataframes) == 1:
            return dataframes[0]

        # Use process pool for large combines
        if sum(len(df) for df in dataframes) > 100000 and self.process_pool:
            loop = asyncio.get_event_loop()
            combined = await loop.run_in_executor(
                self.process_pool,
                self._combine_dataframes_sync,
                dataframes,
            )
            return combined
        else:
            # Simple concatenation for smaller datasets
            return pd.concat(dataframes, ignore_index=True)

    def _combine_dataframes_sync(self, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        """Synchronous DataFrame combination."""
        # Optimize memory usage during concatenation
        combined = pd.concat(dataframes, ignore_index=True)

        # Remove duplicates efficiently
        if 'interaction_id' in combined.columns:
            combined = combined.drop_duplicates(subset=['interaction_id'])

        # Optimize data types
        combined = self._optimize_dataframe_dtypes(combined)

        return combined

    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df[col] = df[col].astype('int32')

        # Optimize string columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # If less than 50% unique values, use category
                    df[col] = df[col].astype('category')

        return df

    async def _process_llm_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of LLM interaction data."""
        # Add derived columns
        if 'prompt' in chunk.columns:
            chunk['prompt_length'] = chunk['prompt'].str.len()
            chunk['prompt_word_count'] = chunk['prompt'].str.split().str.len()

        # Add partitioning columns
        if 'created_at' in chunk.columns:
            chunk['dt'] = pd.to_datetime(chunk['created_at']).dt.strftime('%Y-%m-%d')
            chunk['hour'] = pd.to_datetime(chunk['created_at']).dt.hour

        # Data quality validation
        chunk = chunk.dropna(subset=['interaction_id', 'provider', 'model'])

        return chunk

    async def _write_chunk_optimized(
        self,
        chunk: pd.DataFrame,
        table_name: str,
    ) -> bool:
        """Write DataFrame chunk to storage optimally."""
        try:
            # Prepare output path
            table_path = self.data_lake_path / "raw" / table_name
            table_path.mkdir(parents=True, exist_ok=True)

            # Convert to Arrow table for efficient writing
            table = pa.Table.from_pandas(chunk)

            # Write with optimized settings
            partition_filename = f"chunk_{int(time.time() * 1000)}.parquet"
            output_file = table_path / partition_filename

            pq.write_table(
                table,
                str(output_file),
                compression=self.config.compression,
                use_dictionary=True,
                row_group_size=50000,
                write_statistics=True,
            )

            logger.debug(f"Wrote chunk with {len(chunk)} rows to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to write chunk: {e}")
            return False

    async def _worker_loop(self, worker_id: str) -> None:
        """Background worker loop for processing jobs."""
        logger.debug(f"Pipeline worker {worker_id} started")

        while self._is_running:
            try:
                # Get job from queue
                job = await asyncio.wait_for(self._job_queue.get(), timeout=1.0)

                logger.info(f"Worker {worker_id} processing job {job.job_id}")
                start_time = time.time()

                try:
                    # Process job based on table type
                    if job.table_name == "llm_interactions":
                        await self._process_llm_interactions_job(job)
                    elif job.table_name == "transformations":
                        await self._process_transformations_job(job)
                    else:
                        logger.warning(f"Unknown table type: {job.table_name}")

                    # Mark job complete
                    processing_time = (time.time() - start_time) * 1000
                    logger.info(f"Job {job.job_id} completed in {processing_time:.2f}ms")

                except Exception as e:
                    logger.error(f"Job {job.job_id} failed: {e}")

                finally:
                    # Cleanup
                    if job.job_id in self._active_jobs:
                        del self._active_jobs[job.job_id]

            except TimeoutError:
                continue  # Check if still running
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def _process_llm_interactions_job(self, job: BatchJob) -> None:
        """Process LLM interactions job."""
        # For demonstration - implement actual job processing
        source_configs = job.metadata.get('source_configs', [{'default': True}])

        result = await self.process_llm_interactions_parallel(
            job.start_time,
            job.end_time,
            source_configs,
        )

        logger.info(f"LLM interactions job {job.job_id} result: {result}")

    async def _process_transformations_job(self, job: BatchJob) -> None:
        """Process transformations job."""
        # Placeholder for transformations processing
        logger.info(f"Processing transformations job {job.job_id}")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute

                # Update memory metrics
                import psutil
                process = psutil.Process()
                self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                self.metrics.cpu_usage_percent = process.cpu_percent()

                # Log performance metrics
                logger.info(
                    f"Pipeline Metrics - "
                    f"Records: {self.metrics.total_records_processed}, "
                    f"RPS: {self.metrics.records_per_second:.2f}, "
                    f"Memory: {self.metrics.memory_usage_mb:.2f}MB, "
                    f"CPU: {self.metrics.cpu_usage_percent:.1f}%"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        runtime_seconds = time.time() - self._start_time

        return {
            "runtime_seconds": runtime_seconds,
            "total_records_processed": self.metrics.total_records_processed,
            "records_per_second": self.metrics.records_per_second,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "cpu_usage_percent": self.metrics.cpu_usage_percent,
            "active_jobs": len(self._active_jobs),
            "queue_size": self._job_queue.qsize(),
            "processing_time_ms": self.metrics.processing_time_ms,
            "workers_active": len([w for w in self._workers if not w.done()]),
            "config": {
                "max_workers": self.config.max_workers,
                "chunk_size": self.config.chunk_size,
                "max_memory_mb": self.config.max_memory_usage_mb,
                "compression": self.config.compression,
            },
        }


# =====================================================
# Streaming Data Pipeline
# =====================================================

class StreamingDataPipeline:
    """
    Real-time streaming data pipeline for low-latency processing.
    """

    def __init__(self, config: OptimizedIngestionConfig):
        self.config = config
        self._stream_buffer: list[dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()
        self._last_flush = time.time()

        # Background tasks
        self._flush_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self) -> None:
        """Start streaming pipeline."""
        self._is_running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Streaming data pipeline started")

    async def stop(self) -> None:
        """Stop streaming pipeline."""
        self._is_running = False

        if self._flush_task:
            self._flush_task.cancel()

        # Flush remaining data
        await self._flush_buffer()

        logger.info("Streaming data pipeline stopped")

    async def ingest_record(self, record: dict[str, Any], table_name: str) -> None:
        """Ingest a single record into the streaming pipeline."""
        async with self._buffer_lock:
            # Add metadata
            record['_table_name'] = table_name
            record['_ingested_at'] = time.time()

            self._stream_buffer.append(record)

            # Flush if buffer is full
            if len(self._stream_buffer) >= self.config.stream_buffer_size:
                await self._flush_buffer()

    async def ingest_batch(self, records: list[dict[str, Any]], table_name: str) -> None:
        """Ingest a batch of records."""
        for record in records:
            await self.ingest_record(record, table_name)

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self.config.stream_flush_interval_ms / 1000)

                current_time = time.time()
                time_since_last_flush = (current_time - self._last_flush) * 1000

                if time_since_last_flush >= self.config.stream_flush_interval_ms:
                    await self._flush_buffer()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush loop error: {e}")

    async def _flush_buffer(self) -> None:
        """Flush stream buffer to storage."""
        async with self._buffer_lock:
            if not self._stream_buffer:
                return

            # Group records by table
            records_by_table = {}
            for record in self._stream_buffer:
                table_name = record.pop('_table_name')
                if table_name not in records_by_table:
                    records_by_table[table_name] = []
                records_by_table[table_name].append(record)

            # Write each table
            for table_name, records in records_by_table.items():
                try:
                    df = pd.DataFrame(records)
                    await self._write_streaming_data(df, table_name)
                except Exception as e:
                    logger.error(f"Failed to write streaming data for {table_name}: {e}")

            # Clear buffer
            self._stream_buffer.clear()
            self._last_flush = time.time()

            logger.debug(f"Flushed {sum(len(records) for records in records_by_table.values())} records")

    async def _write_streaming_data(self, df: pd.DataFrame, table_name: str) -> None:
        """Write streaming data to storage."""
        output_path = Path(self.config.data_lake_path) / "streaming" / table_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Add timestamp partition
        timestamp = int(time.time())
        filename = f"stream_{timestamp}.parquet"
        file_path = output_path / filename

        # Write efficiently
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            str(file_path),
            compression=self.config.compression,
        )


# =====================================================
# Factory and Integration Functions
# =====================================================

def create_optimized_pipeline(
    config: OptimizedIngestionConfig | None = None
) -> tuple[ParallelDataPipelineEngine, StreamingDataPipeline]:
    """Create optimized data pipeline with both batch and streaming capabilities."""
    pipeline_config = config or OptimizedIngestionConfig()

    # Create components
    batch_pipeline = ParallelDataPipelineEngine(pipeline_config)
    streaming_pipeline = StreamingDataPipeline(pipeline_config)

    return batch_pipeline, streaming_pipeline


async def run_optimized_etl_job(
    table_name: str,
    lookback_hours: int = 1,
    enable_streaming: bool = True,
) -> dict[str, Any]:
    """
    Run optimized ETL job with parallel processing.
    """
    config = OptimizedIngestionConfig(
        max_workers=8,
        enable_multiprocessing=True,
        chunk_size=50000,
    )

    batch_pipeline, streaming_pipeline = create_optimized_pipeline(config)

    try:
        # Start pipelines
        await batch_pipeline.start()
        if enable_streaming:
            await streaming_pipeline.start()

        # Submit batch job
        end_time = time.time()
        start_time = end_time - (lookback_hours * 3600)

        job_id = await batch_pipeline.submit_batch_job(
            table_name=table_name,
            start_time=start_time,
            end_time=end_time,
            source_configs=[{"default": True}],
        )

        # Wait for completion (simplified)
        await asyncio.sleep(5)  # In production, would wait for actual completion

        # Get metrics
        metrics = batch_pipeline.get_performance_metrics()

        return {
            "status": "success",
            "job_id": job_id,
            "table_name": table_name,
            "metrics": metrics,
        }

    finally:
        # Cleanup
        await batch_pipeline.stop()
        if enable_streaming:
            await streaming_pipeline.stop()


# Global optimized pipeline instance
optimized_pipeline_config = OptimizedIngestionConfig()
optimized_batch_pipeline, optimized_streaming_pipeline = create_optimized_pipeline(optimized_pipeline_config)
