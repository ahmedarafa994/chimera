# Story 1.3: Configuration Persistence System

## Story Information
- **Story ID**: MP-003
- **Epic**: Meta-Prompter Core Infrastructure
- **Priority**: High
- **Points**: 8
- **Sprint**: 1

## User Story
As a **platform administrator**, I want **provider configurations and API keys to persist across restarts** so that **I don't have to reconfigure the system every time**.

## Description
Implement a SQLite-based persistence layer for provider configurations and encrypted API key storage. The system should provide efficient data access patterns with in-memory caching and support transactional operations across multiple repositories.

## Acceptance Criteria

### AC1: SQLite Database Layer
- [x] Async database operations using aiosqlite
- [x] Connection pooling and lifecycle management
- [x] WAL mode enabled for better concurrency
- [x] Configurable database path via environment variable

### AC2: Schema Management
- [x] Version-tracked schema migrations
- [x] Idempotent schema initialization
- [x] Support for rolling migrations
- [x] Tables: provider_configs, api_keys, schema_version

### AC3: Configuration Repository
- [x] Full CRUD operations for provider configurations
- [x] Default provider management per type
- [x] Query by provider type
- [x] Query by name

### AC4: API Key Repository  
- [x] Fernet encryption at rest for all API keys
- [x] Transparent encryption/decryption on storage/retrieval
- [x] Active/inactive key management
- [x] Usage tracking (last_used_at timestamp)
- [x] Unique constraint on (provider_type, key_name)

### AC5: Unit of Work Pattern
- [x] Transaction boundary management
- [x] Atomic operations across repositories
- [x] Automatic rollback on errors
- [x] Context manager interface

### AC6: Configuration Service
- [x] High-level API for configuration operations
- [x] In-memory caching with TTL
- [x] Automatic cache invalidation on writes
- [x] Thread-safe operations via asyncio.Lock

## Technical Specifications

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Service                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  In-Memory Cache                      │   │
│  │     TTL-based, Auto-invalidation on writes           │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Unit of Work                            │
│         Transaction management, Atomic operations            │
├─────────────────────────────────────────────────────────────┤
│    ┌──────────────────┐    ┌──────────────────┐            │
│    │  ConfigRepository │    │ ApiKeyRepository │            │
│    │   CRUD for configs│    │ Encrypted storage│            │
│    └──────────────────┘    └──────────────────┘            │
├─────────────────────────────────────────────────────────────┤
│                   Database Connection                        │
│           aiosqlite, WAL mode, Connection pool              │
├─────────────────────────────────────────────────────────────┤
│                      SQLite Database                         │
│   provider_configs │ api_keys │ schema_version              │
└─────────────────────────────────────────────────────────────┘
```

### Database Schema

#### provider_configs
```sql
CREATE TABLE provider_configs (
    id TEXT PRIMARY KEY,
    provider_type TEXT NOT NULL,
    name TEXT NOT NULL,
    is_default INTEGER NOT NULL DEFAULT 0,
    settings TEXT NOT NULL DEFAULT '{}',  -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_provider_configs_type ON provider_configs(provider_type);
CREATE INDEX idx_provider_configs_default ON provider_configs(is_default);
```

#### api_keys
```sql
CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,
    provider_type TEXT NOT NULL,
    key_name TEXT NOT NULL,
    encrypted_key TEXT NOT NULL,  -- Fernet encrypted
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT,
    UNIQUE(provider_type, key_name)
);

CREATE INDEX idx_api_keys_provider ON api_keys(provider_type);
CREATE INDEX idx_api_keys_active ON api_keys(is_active);
```

### API Reference

#### ConfigurationService Methods

```python
# Provider Configuration
async def create_config(provider_type, name, settings, is_default) -> ProviderConfigEntity
async def get_config(config_id) -> Optional[ProviderConfigEntity]
async def get_all_configs() -> list[ProviderConfigEntity]
async def get_configs_by_provider(provider_type) -> list[ProviderConfigEntity]
async def get_default_config(provider_type) -> Optional[ProviderConfigEntity]
async def update_config(config_id, name, settings, is_default) -> ProviderConfigEntity
async def delete_config(config_id) -> bool
async def set_default_config(config_id) -> ProviderConfigEntity

# API Key Management
async def store_api_key(provider_type, key_name, api_key) -> ApiKeyEntity
async def get_api_key(key_id) -> Optional[ApiKeyEntity]
async def get_active_api_key(provider_type) -> Optional[ApiKeyEntity]
async def rotate_api_key(key_id, new_api_key) -> ApiKeyEntity
async def deactivate_api_key(key_id) -> bool
async def delete_api_key(key_id) -> bool
async def record_key_usage(key_id) -> bool

# Cache Management
def clear_cache() -> None
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CHIMERA_DB_PATH` | Path to SQLite database | `data/chimera.db` |
| `CHIMERA_ENCRYPTION_KEY` | Base64 encryption key | Auto-generated |
| `CHIMERA_ENCRYPTION_PASSWORD` | Password for key derivation | Default password |

### File Structure

```
backend-api/app/
├── infrastructure/
│   ├── database/
│   │   ├── __init__.py          # Module exports
│   │   ├── connection.py        # DatabaseConnection class
│   │   ├── schema.py            # Schema migrations
│   │   └── unit_of_work.py      # UoW pattern
│   └── repositories/
│       ├── __init__.py          # Repository exports
│       ├── base.py              # BaseRepository ABC
│       ├── config_repository.py # Provider configs
│       └── api_key_repository.py # Encrypted keys
├── domain/
│   └── services/
│       ├── __init__.py
│       └── config_service.py    # Service layer
└── core/
    └── encryption.py            # Fernet encryption (existing)

backend-api/tests/
└── test_config_persistence.py   # Comprehensive test suite
```

## Dependencies
- **Story 1.1**: Uses encryption module from Provider Configuration Management
- **aiosqlite**: Async SQLite library
- **cryptography**: Fernet encryption (already installed)

## Security Considerations
1. API keys encrypted at rest using Fernet (AES-128-CBC with HMAC)
2. Encryption key derived from password using PBKDF2 (100,000 iterations)
3. Keys decrypted only when needed for runtime use
4. Database file permissions should be restricted in production

## Testing

### Test Categories
1. **Database Connection**: Initialize, execute, transactions
2. **Schema Management**: Table creation, migrations, versions
3. **Config Repository**: CRUD, filtering, default management
4. **API Key Repository**: Encryption, decryption, deactivation
5. **Unit of Work**: Transactions, rollback, repository access
6. **Configuration Service**: Caching, invalidation, high-level ops

### Running Tests
```bash
cd backend-api
poetry run pytest tests/test_config_persistence.py -v
```

## Definition of Done
- [x] All acceptance criteria met
- [x] Unit tests written and passing
- [x] Encryption verified for API keys
- [x] Cache invalidation working correctly
- [x] Documentation updated
- [x] Code follows project style guidelines

## Implementation Notes

### Caching Strategy
- Default TTL: 5 minutes
- Namespace-based cache keys: `config:{id}`, `api_key:{id}`
- Full invalidation on write operations
- Thread-safe via asyncio.Lock

### Error Handling
- `ConfigServiceError`: Base service exception
- `ConfigNotFoundError`: Entity not found
- `DuplicateEntityError`: Duplicate key constraint
- `EntityNotFoundError`: Repository-level not found
- `DatabaseError`: Connection/query failures

### Performance Considerations
- WAL mode for concurrent reads during writes
- Connection reuse via singleton pattern
- Lazy repository instantiation in UoW
- Bulk operations supported via executemany

## Related Stories
- **Story 1.1**: Provider Configuration Management (dependency)
- **Story 1.2**: Direct API Integration (uses stored configs)
- **Story 1.4**: Provider Health Monitoring (will use persistence)