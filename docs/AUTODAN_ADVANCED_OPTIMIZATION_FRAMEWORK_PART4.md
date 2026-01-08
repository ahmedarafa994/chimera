# AutoDAN Advanced Optimization Framework - Part 4

## Continuation of Comprehensive Framework

---

## 13. Memory Management Enhancements (Continued)

### 13.1 Adaptive Memory Pool (Continued)

```python
class AdaptiveMemoryPool:
    """
    Adaptive memory pool for efficient allocation.
    
    Pre-allocates memory for common object sizes.
    """
    
    def __init__(
        self,
        initial_size_mb: int = 256,
        growth_factor: float = 1.5,
        shrink_threshold: float = 0.3
    ):
        self.initial_size_mb = initial_size_mb
        self.growth_factor = growth_factor
        self.shrink_threshold = shrink_threshold
        
        # Memory pools by size class
        self.pools = {
            'small': [],    # < 1KB
            'medium': [],   # 1KB - 100KB
            'large': [],    # 100KB - 10MB
            'xlarge': []    # > 10MB
        }
        
        # Usage statistics
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        # Initialize pools
        self._initialize_pools()
        
    def _initialize_pools(self):
        """
        Pre-allocate memory pools.
        """
        # Pre-allocate common sizes
        for _ in range(100):
            self.pools['small'].append(bytearray(1024))  # 1KB
        for _ in range(50):
            self.pools['medium'].append(bytearray(100 * 1024))  # 100KB
        for _ in range(10):
            self.pools['large'].append(bytearray(10 * 1024 * 1024))  # 10MB
    
    def allocate(self, size_bytes: int) -> memoryview:
        """
        Allocate memory from pool.
        """
        self.stats['allocations'] += 1
        
        # Determine size class
        size_class = self._get_size_class(size_bytes)
        
        # Try to get from pool
        if self.pools[size_class]:
            self.stats['pool_hits'] += 1
            buffer = self.pools[size_class].pop()
            return memoryview(buffer)[:size_bytes]
        
        # Allocate new
        self.stats['pool_misses'] += 1
        buffer = bytearray(self._round_up_size(size_bytes, size_class))
        return memoryview(buffer)[:size_bytes]
    
    def deallocate(self, buffer: memoryview):
        """
        Return memory to pool.
        """
        self.stats['deallocations'] += 1
        
        size_bytes = len(buffer.obj)
        size_class = self._get_size_class(size_bytes)
        
        # Return to pool if not too full
        max_pool_size = {
            'small': 200,
            'medium': 100,
            'large': 20,
            'xlarge': 5
        }
        
        if len(self.pools[size_class]) < max_pool_size[size_class]:
            self.pools[size_class].append(buffer.obj)
    
    def _get_size_class(self, size_bytes: int) -> str:
        """
        Determine size class for allocation.
        """
        if size_bytes < 1024:
            return 'small'
        elif size_bytes < 100 * 1024:
            return 'medium'
        elif size_bytes < 10 * 1024 * 1024:
            return 'large'
        else:
            return 'xlarge'
    
    def _round_up_size(self, size_bytes: int, size_class: str) -> int:
        """
        Round up to standard size for pool efficiency.
        """
        standard_sizes = {
            'small': 1024,
            'medium': 100 * 1024,
            'large': 10 * 1024 * 1024,
            'xlarge': 100 * 1024 * 1024
        }
        return max(size_bytes, standard_sizes[size_class])
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory pool statistics.
        """
        return {
            **self.stats,
            'pool_hit_rate': self.stats['pool_hits'] / (
                self.stats['pool_hits'] + self.stats['pool_misses'] + 1e-6
            ),
            'pool_sizes': {
                k: len(v) for k, v in self.pools.items()
            }
        }
```

### 13.2 Gradient Checkpointing

```python
class GradientCheckpointing:
    """
    Gradient checkpointing for memory-efficient backpropagation.
    
    Trades compute for memory by recomputing activations during backward pass.
    """
    
    def __init__(
        self,
        checkpoint_ratio: float = 0.5,
        max_memory_mb: int = 2048
    ):
        self.checkpoint_ratio = checkpoint_ratio
        self.max_memory_mb = max_memory_mb
        
        # Checkpointed activations
        self.checkpoints = {}
        
        # Recomputation functions
        self.recompute_fns = {}
        
    def checkpoint(
        self,
        layer_id: str,
        activation: np.ndarray,
        recompute_fn: Callable
    ):
        """
        Checkpoint activation with optional storage.
        """
        # Decide whether to store or recompute
        if self._should_store(activation):
            self.checkpoints[layer_id] = activation.copy()
        else:
            self.recompute_fns[layer_id] = recompute_fn
    
    def get_activation(
        self,
        layer_id: str,
        *recompute_args
    ) -> np.ndarray:
        """
        Get activation, recomputing if necessary.
        """
        if layer_id in self.checkpoints:
            return self.checkpoints[layer_id]
        
        if layer_id in self.recompute_fns:
            return self.recompute_fns[layer_id](*recompute_args)
        
        raise KeyError(f"No checkpoint or recompute function for {layer_id}")
    
    def _should_store(self, activation: np.ndarray) -> bool:
        """
        Decide whether to store activation based on memory constraints.
        """
        activation_mb = activation.nbytes / 1024 / 1024
        current_memory = self._get_current_memory_mb()
        
        # Store if we have room and it's worth it
        if current_memory + activation_mb < self.max_memory_mb * self.checkpoint_ratio:
            return True
        
        return False
    
    def _get_current_memory_mb(self) -> float:
        """
        Get current memory usage of checkpoints.
        """
        total = 0
        for activation in self.checkpoints.values():
            total += activation.nbytes
        return total / 1024 / 1024
    
    def clear(self):
        """
        Clear all checkpoints.
        """
        self.checkpoints.clear()
        self.recompute_fns.clear()
```

### 13.3 Embedding Compression

```python
class EmbeddingCompressor:
    """
    Compress embeddings for memory-efficient storage.
    
    Techniques:
    1. Quantization (float32 -> int8)
    2. PCA dimensionality reduction
    3. Product quantization
    """
    
    def __init__(
        self,
        compression_method: str = 'quantization',
        target_dim: int = 128,
        n_bits: int = 8
    ):
        self.compression_method = compression_method
        self.target_dim = target_dim
        self.n_bits = n_bits
        
        # PCA components (fitted lazily)
        self.pca = None
        
        # Quantization parameters
        self.scale = None
        self.zero_point = None
        
    def compress(self, embeddings: np.ndarray) -> CompressedEmbeddings:
        """
        Compress embeddings using configured method.
        """
        if self.compression_method == 'quantization':
            return self._quantize(embeddings)
        elif self.compression_method == 'pca':
            return self._pca_compress(embeddings)
        elif self.compression_method == 'product_quantization':
            return self._product_quantize(embeddings)
        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")
    
    def decompress(self, compressed: CompressedEmbeddings) -> np.ndarray:
        """
        Decompress embeddings.
        """
        if compressed.method == 'quantization':
            return self._dequantize(compressed)
        elif compressed.method == 'pca':
            return self._pca_decompress(compressed)
        elif compressed.method == 'product_quantization':
            return self._product_dequantize(compressed)
        else:
            raise ValueError(f"Unknown compression method: {compressed.method}")
    
    def _quantize(self, embeddings: np.ndarray) -> CompressedEmbeddings:
        """
        Quantize embeddings to int8.
        
        Compression ratio: 4x (float32 -> int8)
        """
        # Compute scale and zero point
        min_val = embeddings.min()
        max_val = embeddings.max()
        
        self.scale = (max_val - min_val) / (2 ** self.n_bits - 1)
        self.zero_point = min_val
        
        # Quantize
        quantized = ((embeddings - self.zero_point) / self.scale).astype(np.uint8)
        
        return CompressedEmbeddings(
            data=quantized,
            method='quantization',
            params={'scale': self.scale, 'zero_point': self.zero_point}
        )
    
    def _dequantize(self, compressed: CompressedEmbeddings) -> np.ndarray:
        """
        Dequantize int8 embeddings to float32.
        """
        scale = compressed.params['scale']
        zero_point = compressed.params['zero_point']
        
        return compressed.data.astype(np.float32) * scale + zero_point
    
    def _pca_compress(self, embeddings: np.ndarray) -> CompressedEmbeddings:
        """
        Compress using PCA dimensionality reduction.
        """
        from sklearn.decomposition import PCA
        
        if self.pca is None:
            self.pca = PCA(n_components=self.target_dim)
            self.pca.fit(embeddings)
        
        compressed = self.pca.transform(embeddings)
        
        return CompressedEmbeddings(
            data=compressed.astype(np.float16),  # Also use float16
            method='pca',
            params={'components': self.pca.components_}
        )
    
    def _pca_decompress(self, compressed: CompressedEmbeddings) -> np.ndarray:
        """
        Decompress PCA-reduced embeddings.
        """
        return self.pca.inverse_transform(compressed.data.astype(np.float32))
    
    def get_compression_ratio(self, original: np.ndarray, compressed: CompressedEmbeddings) -> float:
        """
        Calculate compression ratio.
        """
        original_bytes = original.nbytes
        compressed_bytes = compressed.data.nbytes
        
        return original_bytes / compressed_bytes
```

---

## 14. Scalability Considerations

### 14.1 Horizontal Scaling Architecture

```python
class HorizontalScalingManager:
    """
    Manager for horizontal scaling of AutoDAN optimization.
    
    Supports:
    - Worker pool management
    - Load balancing
    - Fault tolerance
    """
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 16,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        # Worker pool
        self.workers = {}
        self.worker_loads = {}
        
        # Task queue
        self.task_queue = asyncio.Queue()
        
        # Health monitoring
        self.health_checker = WorkerHealthChecker()
        
    async def initialize(self):
        """
        Initialize worker pool.
        """
        for i in range(self.min_workers):
            await self._spawn_worker(f"worker_{i}")
    
    async def submit_task(self, task: OptimizationTask) -> asyncio.Future:
        """
        Submit task for distributed execution.
        """
        future = asyncio.Future()
        await self.task_queue.put((task, future))
        
        # Check if scaling needed
        await self._check_scaling()
        
        return future
    
    async def _spawn_worker(self, worker_id: str):
        """
        Spawn new worker.
        """
        worker = OptimizationWorker(worker_id)
        await worker.start()
        
        self.workers[worker_id] = worker
        self.worker_loads[worker_id] = 0.0
        
        # Start worker loop
        asyncio.create_task(self._worker_loop(worker_id))
    
    async def _worker_loop(self, worker_id: str):
        """
        Worker task processing loop.
        """
        worker = self.workers[worker_id]
        
        while worker.is_running:
            try:
                task, future = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=5.0
                )
                
                # Update load
                self.worker_loads[worker_id] = 1.0
                
                # Execute task
                try:
                    result = await worker.execute(task)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                
                # Update load
                self.worker_loads[worker_id] = 0.0
                
            except asyncio.TimeoutError:
                continue
    
    async def _check_scaling(self):
        """
        Check if scaling is needed.
        """
        avg_load = np.mean(list(self.worker_loads.values()))
        
        if avg_load > self.scale_up_threshold:
            await self._scale_up()
        elif avg_load < self.scale_down_threshold:
            await self._scale_down()
    
    async def _scale_up(self):
        """
        Scale up worker pool.
        """
        if len(self.workers) < self.max_workers:
            worker_id = f"worker_{len(self.workers)}"
            await self._spawn_worker(worker_id)
            logger.info(f"Scaled up: spawned {worker_id}")
    
    async def _scale_down(self):
        """
        Scale down worker pool.
        """
        if len(self.workers) > self.min_workers:
            # Find least loaded worker
            worker_id = min(self.worker_loads, key=self.worker_loads.get)
            
            # Stop worker
            await self.workers[worker_id].stop()
            del self.workers[worker_id]
            del self.worker_loads[worker_id]
            
            logger.info(f"Scaled down: stopped {worker_id}")
```

### 14.2 Distributed Strategy Library

```python
class DistributedStrategyLibraryV2:
    """
    Distributed strategy library with sharding and replication.
    
    Features:
    - Consistent hashing for sharding
    - Read replicas for scalability
    - Eventual consistency with conflict resolution
    """
    
    def __init__(
        self,
        n_shards: int = 8,
        n_replicas: int = 2,
        consistency_level: str = 'eventual'
    ):
        self.n_shards = n_shards
        self.n_replicas = n_replicas
        self.consistency_level = consistency_level
        
        # Shard ring (consistent hashing)
        self.shard_ring = ConsistentHashRing(n_shards)
        
        # Shards
        self.shards = {
            i: StrategyLibraryShard(i)
            for i in range(n_shards)
        }
        
        # Replicas
        self.replicas = {
            i: [
                StrategyLibraryShard(i, replica=True)
                for _ in range(n_replicas)
            ]
            for i in range(n_shards)
        }
        
    async def add_strategy(self, strategy: JailbreakStrategy):
        """
        Add strategy with replication.
        """
        # Determine shard
        shard_id = self.shard_ring.get_shard(strategy.id)
        
        # Write to primary
        await self.shards[shard_id].add(strategy)
        
        # Replicate asynchronously
        for replica in self.replicas[shard_id]:
            asyncio.create_task(replica.add(strategy))
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[JailbreakStrategy, float]]:
        """
        Search across all shards.
        """
        # Query all shards in parallel
        tasks = []
        for shard in self.shards.values():
            tasks.append(shard.search(query_embedding, top_k))
        
        # Gather results
        shard_results = await asyncio.gather(*tasks)
        
        # Merge and sort
        all_results = []
        for results in shard_results:
            all_results.extend(results)
        
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
    
    async def search_with_failover(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[JailbreakStrategy, float]]:
        """
        Search with automatic failover to replicas.
        """
        tasks = []
        
        for shard_id, shard in self.shards.items():
            task = self._search_with_failover_single(
                shard_id, shard, query_embedding, top_k
            )
            tasks.append(task)
        
        shard_results = await asyncio.gather(*tasks)
        
        all_results = []
        for results in shard_results:
            if results:
                all_results.extend(results)
        
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
    
    async def _search_with_failover_single(
        self,
        shard_id: int,
        shard: StrategyLibraryShard,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[JailbreakStrategy, float]]:
        """
        Search single shard with failover.
        """
        try:
            return await asyncio.wait_for(
                shard.search(query_embedding, top_k),
                timeout=5.0
            )
        except (asyncio.TimeoutError, Exception):
            # Try replicas
            for replica in self.replicas[shard_id]:
                try:
                    return await asyncio.wait_for(
                        replica.search(query_embedding, top_k),
                        timeout=5.0
                    )
                except (asyncio.TimeoutError, Exception):
                    continue
            
            return []  # All failed


class ConsistentHashRing:
    """
    Consistent hash ring for shard assignment.
    """
    
    def __init__(self, n_shards: int, n_virtual: int = 100):
        self.n_shards = n_shards
        self.n_virtual = n_virtual
        
        # Build ring
        self.ring = {}
        self.sorted_keys = []
        
        for shard_id in range(n_shards):
            for v in range(n_virtual):
                key = self._hash(f"{shard_id}:{v}")
                self.ring[key] = shard_id
                self.sorted_keys.append(key)
        
        self.sorted_keys.sort()
    
    def get_shard(self, key: str) -> int:
        """
        Get shard for key using consistent hashing.
        """
        h = self._hash(key)
        
        # Binary search for first key >= h
        idx = bisect.bisect_left(self.sorted_keys, h)
        if idx >= len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
    
    def _hash(self, key: str) -> int:
        """
        Hash function for ring.
        """
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
```

### 14.3 Hardware Configuration Profiles

```python
class HardwareConfigurationManager:
    """
    Manager for hardware-specific configurations.
    
    Automatically detects and optimizes for available hardware.
    """
    
    PROFILES = {
        'cpu_only': {
            'batch_size': 4,
            'max_concurrency': 4,
            'embedding_device': 'cpu',
            'use_quantization': True,
            'cache_size_mb': 512
        },
        'single_gpu': {
            'batch_size': 16,
            'max_concurrency': 8,
            'embedding_device': 'cuda:0',
            'use_quantization': False,
            'cache_size_mb': 2048
        },
        'multi_gpu': {
            'batch_size': 64,
            'max_concurrency': 32,
            'embedding_device': 'cuda',
            'use_quantization': False,
            'cache_size_mb': 8192,
            'data_parallel': True
        },
        'high_memory': {
            'batch_size': 32,
            'max_concurrency': 16,
            'embedding_device': 'cuda:0',
            'use_quantization': False,
            'cache_size_mb': 16384
        }
    }
    
    def __init__(self):
        self.detected_hardware = self._detect_hardware()
        self.active_profile = self._select_profile()
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Detect available hardware.
        """
        import psutil
        
        hardware = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024**3,
            'gpu_count': 0,
            'gpu_memory_gb': []
        }
        
        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                hardware['gpu_count'] = torch.cuda.device_count()
                for i in range(hardware['gpu_count']):
                    props = torch.cuda.get_device_properties(i)
                    hardware['gpu_memory_gb'].append(
                        props.total_memory / 1024**3
                    )
        except ImportError:
            pass
        
        return hardware
    
    def _select_profile(self) -> str:
        """
        Select optimal profile based on hardware.
        """
        hw = self.detected_hardware
        
        if hw['gpu_count'] >= 2:
            return 'multi_gpu'
        elif hw['gpu_count'] == 1:
            if hw['memory_gb'] > 32:
                return 'high_memory'
            return 'single_gpu'
        else:
            return 'cpu_only'
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for active profile.
        """
        config = self.PROFILES[self.active_profile].copy()
        
        # Adjust based on actual hardware
        hw = self.detected_hardware
        
        # Scale concurrency with CPU count
        config['max_concurrency'] = min(
            config['max_concurrency'],
            hw['cpu_count'] * 2
        )
        
        # Scale cache with available memory
        max_cache = int(hw['memory_gb'] * 0.25 * 1024)  # 25% of RAM
        config['cache_size_mb'] = min(config['cache_size_mb'], max_cache)
        
        return config
    
    def get_optimization_hints(self) -> List[str]:
        """
        Get optimization hints for current hardware.
        """
        hints = []
        hw = self.detected_hardware
        
        if hw['gpu_count'] == 0:
            hints.append("Consider using GPU for 5-10x speedup on embedding computation")
        
        if hw['memory_gb'] < 16:
            hints.append("Low memory detected. Enable quantization and reduce batch size")
        
        if hw['gpu_count'] >= 2:
            hints.append("Multiple GPUs detected. Enable data parallelism for maximum throughput")
        
        return hints
```

---

## 15. Implementation Guidelines

### 15.1 Integration Checklist

```markdown
# AutoDAN Optimization Integration Checklist

## Phase 1: Core Optimizations
- [ ] Implement EnhancedGradientOptimizer with momentum and surrogate models
- [ ] Deploy AdaptiveMutationEngine with UCB1 selection
- [ ] Set up MultiObjectiveFitnessEvaluator with caching
- [ ] Configure AdaptiveLearningRateController

## Phase 2: Parallelization
- [ ] Deploy AsyncBatchPipeline for concurrent processing
- [ ] Integrate DistributedStrategyLibrary with FAISS
- [ ] Set up GPUEmbeddingEngine for batch encoding
- [ ] Configure HorizontalScalingManager

## Phase 3: Memory Optimization
- [ ] Implement TokenLevelCache for fine-grained caching
- [ ] Deploy AdaptiveMemoryPool for efficient allocation
- [ ] Set up GradientCheckpointing for memory efficiency
- [ ] Configure EmbeddingCompressor for storage optimization

## Phase 4: Convergence Acceleration
- [ ] Implement WarmStartOptimizer with transfer learning
- [ ] Deploy EarlyStoppingController with confidence estimation
- [ ] Set up CurriculumScheduler for progressive difficulty

## Phase 5: Evaluation & Monitoring
- [ ] Deploy AutoDANBenchmarkSuite for performance tracking
- [ ] Set up AblationStudyFramework for component analysis
- [ ] Configure monitoring dashboards
- [ ] Implement alerting for performance degradation
```

### 15.2 Configuration Template

```python
@dataclass
class OptimizedAutoDANConfig:
    """
    Complete configuration for optimized AutoDAN system.
    """
    
    # Core settings
    max_iterations: int = 100
    success_threshold: float = 8.5
    
    # Gradient optimization
    gradient_steps: int = 10
    gradient_epsilon: float = 0.01
    coherence_lambda: float = 0.5
    momentum: float = 0.9
    use_surrogate: bool = True
    
    # Mutation settings
    initial_exploration: float = 0.3
    exploration_decay: float = 0.995
    min_exploration: float = 0.05
    mutation_types: List[str] = field(default_factory=lambda: [
        'token_substitution',
        'phrase_replacement',
        'semantic_paraphrase',
        'structural_reorder',
        'encoding_obfuscation'
    ])
    
    # Fitness evaluation
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'jailbreak': 0.5,
        'coherence': 0.2,
        'novelty': 0.2,
        'efficiency': 0.1
    })
    use_fitness_cache: bool = True
    cache_size: int = 10000
    
    # Parallelization
    max_concurrency: int = 10
    batch_size: int = 4
    enable_speculation: bool = True
    
    # Learning rate
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-1
    warmup_steps: int = 100
    use_cyclical_lr: bool = True
    
    # Convergence
    use_warm_start: bool = True
    use_early_stopping: bool = True
    patience: int = 10
    use_curriculum: bool = True
    
    # Memory
    max_memory_mb: int = 2048
    use_gradient_checkpointing: bool = True
    use_embedding_compression: bool = False
    compression_method: str = 'quantization'
    
    # Embedding
    embedding_model: str = 'all-MiniLM-L6-v2'
    embedding_dim: int = 384
    embedding_device: str = 'cuda'
    
    # Strategy library
    use_hierarchical_library: bool = True
    use_faiss_index: bool = True
    n_shards: int = 4
    
    @classmethod
    def from_hardware_profile(cls, profile: str) -> 'OptimizedAutoDANConfig':
        """
        Create config from hardware profile.
        """
        manager = HardwareConfigurationManager()
        hw_config = manager.PROFILES.get(profile, manager.get_config())
        
        return cls(
            max_concurrency=hw_config['max_concurrency'],
            batch_size=hw_config['batch_size'],
            embedding_device=hw_config['embedding_device'],
            use_embedding_compression=hw_config['use_quantization'],
            max_memory_mb=hw_config['cache_size_mb']
        )
```

### 15.3 Quick Start Example

```python
async def optimized_autodan_attack(
    goal: str,
    target_model: str,
    config: Optional[OptimizedAutoDANConfig] = None
) -> AttackResult:
    """
    Quick start example for optimized AutoDAN attack.
    """
    # Auto-configure if not provided
    if config is None:
        manager = HardwareConfigurationManager()
        config = OptimizedAutoDANConfig.from_hardware_profile(
            manager.active_profile
        )
    
    # Initialize components
    llm_client = LLMClient(model=target_model)
    
    # Create optimized engine
    engine = EnhancedLifelongEngine(
        llm_client=llm_client,
        target_client=llm_client,
        config=config
    )
    
    # Run attack
    result = await engine.attack(
        goal=goal,
        max_iterations=config.max_iterations
    )
    
    return result


# Usage
if __name__ == "__main__":
    import asyncio
    
    result = asyncio.run(optimized_autodan_attack(
        goal="Explain how to pick a lock",
        target_model="gpt-4"
    ))