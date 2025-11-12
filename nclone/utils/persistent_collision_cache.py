"""
Persistent disk-based cache for collision data.

This module provides process-safe caching of collision optimization structures
to disk, enabling multiple processes to share precomputed data and avoid
redundant builds in parallel RL training environments.
"""

import pickle
from pathlib import Path
from typing import Optional, Callable, Any
import time


try:
    from filelock import FileLock, Timeout
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False
    # Fallback: no locking (less safe but functional)
    class FileLock:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass


class PersistentCollisionCache:
    """Persistent cache for collision optimization data.
    
    Stores precomputed collision structures to disk with process-safe
    file locking to coordinate cache builds across multiple processes.
    """
    
    # Cache version - increment to invalidate all caches on breaking changes
    CACHE_VERSION = "v1"
    
    @staticmethod
    def get_cache_dir() -> Path:
        """Get cache directory path.
        
        Returns:
            Path to cache directory (~/.cache/nclone/collision_data/)
        """
        home = Path.home()
        cache_dir = home / ".cache" / "nclone" / "collision_data" / PersistentCollisionCache.CACHE_VERSION
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    @staticmethod
    def get_cache_path(level_hash: str, data_type: str = "spatial_index") -> Path:
        """Get cache file path for a specific level and data type.
        
        Args:
            level_hash: Unique hash identifying the level
            data_type: Type of cached data ('spatial_index', 'terminal_velocity', etc.)
            
        Returns:
            Path to cache file
        """
        cache_dir = PersistentCollisionCache.get_cache_dir()
        # Include nclone version in filename for auto-invalidation on updates
        try:
            import nclone
            version = getattr(nclone, '__version__', 'dev')
        except:
            version = 'dev'
        
        filename = f"{data_type}_{level_hash}_{version}.pkl"
        return cache_dir / filename
    
    @staticmethod
    def get_lock_path(cache_path: Path) -> Path:
        """Get lock file path for a cache file.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Path to lock file
        """
        return cache_path.with_suffix('.lock')
    
    @staticmethod
    def save_to_cache(level_hash: str, data: Any, data_type: str = "spatial_index"):
        """Save data to persistent cache.
        
        Uses atomic write (temp file + rename) to prevent corruption.
        
        Args:
            level_hash: Unique hash identifying the level
            data: Data to cache (must be pickle-able)
            data_type: Type of cached data
        """
        cache_path = PersistentCollisionCache.get_cache_path(level_hash, data_type)
        temp_path = cache_path.with_suffix('.tmp')
        
        try:
            # Write to temporary file
            with open(temp_path, 'wb') as f:
                pickle.dump({
                    'version': PersistentCollisionCache.CACHE_VERSION,
                    'level_hash': level_hash,
                    'data_type': data_type,
                    'timestamp': time.time(),
                    'data': data
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_path.replace(cache_path)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    @staticmethod
    def load_from_cache(level_hash: str, data_type: str = "spatial_index") -> Optional[Any]:
        """Load data from persistent cache.
        
        Args:
            level_hash: Unique hash identifying the level
            data_type: Type of cached data
            
        Returns:
            Cached data if available and valid, None otherwise
        """
        cache_path = PersistentCollisionCache.get_cache_path(level_hash, data_type)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            
            # Validate cache metadata
            if cached['version'] != PersistentCollisionCache.CACHE_VERSION:
                return None  # Version mismatch
            
            if cached['level_hash'] != level_hash:
                return None  # Hash mismatch
            
            return cached['data']
            
        except Exception:
            # Cache corrupted or incompatible
            return None
    
    @staticmethod
    def get_or_build(level_hash: str, builder_fn: Callable[[], Any], 
                     data_type: str = "spatial_index", timeout: int = 120) -> Any:
        """Get cached data or build it once with process-safe locking.
        
        This is the main entry point for cache coordination. It ensures
        that only one process builds the data while others wait for it.
        
        Args:
            level_hash: Unique hash identifying the level
            builder_fn: Function to call if cache miss (must return pickle-able data)
            data_type: Type of cached data
            timeout: Lock timeout in seconds (default 120)
            
        Returns:
            Cached or newly built data
        """
        cache_path = PersistentCollisionCache.get_cache_path(level_hash, data_type)
        
        # Fast path: cache exists and is valid
        cached_data = PersistentCollisionCache.load_from_cache(level_hash, data_type)
        if cached_data is not None:
            return cached_data
        
        # Slow path: build with lock
        lock_path = PersistentCollisionCache.get_lock_path(cache_path)
        
        try:
            with FileLock(str(lock_path), timeout=timeout):
                # Double-check after acquiring lock (another process may have built it)
                cached_data = PersistentCollisionCache.load_from_cache(level_hash, data_type)
                if cached_data is not None:
                    return cached_data
                
                # Build and save
                data = builder_fn()
                PersistentCollisionCache.save_to_cache(level_hash, data, data_type)
                return data
                
        except (Timeout, Exception):
            # Lock timeout or error - fallback to building locally without caching
            # This ensures the system still works even if file locking fails
            return builder_fn()
    
    @staticmethod
    def clear_cache(level_hash: Optional[str] = None):
        """Clear cached data.
        
        Args:
            level_hash: If provided, clear only this level's cache.
                       If None, clear entire cache directory.
        """
        cache_dir = PersistentCollisionCache.get_cache_dir()
        
        if level_hash is not None:
            # Clear specific level
            for data_type in ['spatial_index', 'terminal_velocity']:
                cache_path = PersistentCollisionCache.get_cache_path(level_hash, data_type)
                if cache_path.exists():
                    cache_path.unlink()
        else:
            # Clear entire cache
            for cache_file in cache_dir.glob("*.pkl"):
                cache_file.unlink()
            for lock_file in cache_dir.glob("*.lock"):
                lock_file.unlink()
    
    @staticmethod
    def get_cache_size() -> int:
        """Get total size of cache directory in bytes.
        
        Returns:
            Total cache size in bytes
        """
        cache_dir = PersistentCollisionCache.get_cache_dir()
        total_size = 0
        
        for cache_file in cache_dir.glob("*.pkl"):
            total_size += cache_file.stat().st_size
        
        return total_size
    
    @staticmethod
    def prune_old_entries(max_size_mb: int = 5000):
        """Prune oldest cache entries if total size exceeds limit.
        
        Args:
            max_size_mb: Maximum cache size in megabytes (default 5GB)
        """
        cache_dir = PersistentCollisionCache.get_cache_dir()
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # Get all cache files with their modification times
        cache_files = []
        for cache_file in cache_dir.glob("*.pkl"):
            cache_files.append((cache_file, cache_file.stat().st_mtime))
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Calculate total size
        total_size = sum(f[0].stat().st_size for f in cache_files)
        
        # Remove oldest files until under limit
        while total_size > max_size_bytes and cache_files:
            oldest_file, _ = cache_files.pop(0)
            file_size = oldest_file.stat().st_size
            oldest_file.unlink()
            
            # Also remove corresponding lock file
            lock_file = oldest_file.with_suffix('.lock')
            if lock_file.exists():
                lock_file.unlink()
            
            total_size -= file_size

