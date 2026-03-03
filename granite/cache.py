import hashlib
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import pandas as pd

class AccessibilityCache:
    """
    Tracks accessibility computations and implements basic caching.
    Logs all access patterns for workload analysis.
    """
    
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Separate directories for different cache types
        self.absolute_cache = self.cache_dir / 'absolute'
        self.differential_cache = self.cache_dir / 'differential'
        self.absolute_cache.mkdir(exist_ok=True)
        self.differential_cache.mkdir(exist_ok=True)
        
        # Log file for workload analysis
        self.log_file = self.cache_dir / 'access_log.jsonl'
        
    def _hash_key(self, **kwargs) -> str:
        """Create deterministic hash from parameters."""
        key_str = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _log_access(self, operation: str, cache_type: str, key: str, 
                    hit: bool, metadata: Dict = None):
        """Log every cache access for workload analysis."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'cache_type': cache_type,
            'key': key,
            'hit': hit,
            'metadata': metadata or {}
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_absolute(self, mode: str, dest_type: str, threshold: int,
                     origins_hash: str) -> Optional[Any]:
        """Retrieve absolute accessibility result."""
        key = self._hash_key(mode=mode, dest_type=dest_type, 
                            threshold=threshold, origins=origins_hash)
        cache_file = self.absolute_cache / f"{key}.pkl"
        
        if cache_file.exists():
            self._log_access('get', 'absolute', key, hit=True,
                           metadata={'mode': mode, 'dest_type': dest_type, 
                                   'threshold': threshold})
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        self._log_access('get', 'absolute', key, hit=False,
                        metadata={'mode': mode, 'dest_type': dest_type, 
                                'threshold': threshold})
        return None
    
    def set_absolute(self, value: Any, mode: str, dest_type: str, 
                     threshold: int, origins_hash: str):
        """Store absolute accessibility result."""
        key = self._hash_key(mode=mode, dest_type=dest_type,
                            threshold=threshold, origins=origins_hash)
        cache_file = self.absolute_cache / f"{key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
        
        self._log_access('set', 'absolute', key, hit=True,
                        metadata={'mode': mode, 'dest_type': dest_type,
                                'threshold': threshold})
    
    def get_differential(self, mode_a: str, mode_b: str, dest_type: str,
                        threshold: int, origins_hash: str, 
                        operation: str = 'ratio') -> Optional[Any]:
        """Retrieve differential (ratio or difference) result."""
        key = self._hash_key(mode_a=mode_a, mode_b=mode_b, dest_type=dest_type,
                            threshold=threshold, origins=origins_hash,
                            operation=operation)
        cache_file = self.differential_cache / f"{key}.pkl"
        
        if cache_file.exists():
            self._log_access('get', 'differential', key, hit=True,
                           metadata={'modes': f"{mode_a}_vs_{mode_b}",
                                   'operation': operation})
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        self._log_access('get', 'differential', key, hit=False,
                        metadata={'modes': f"{mode_a}_vs_{mode_b}",
                                'operation': operation})
        return None
    
    def set_differential(self, value: Any, mode_a: str, mode_b: str,
                        dest_type: str, threshold: int, origins_hash: str,
                        operation: str = 'ratio'):
        """Store differential (ratio or difference) result."""
        key = self._hash_key(mode_a=mode_a, mode_b=mode_b, dest_type=dest_type,
                            threshold=threshold, origins=origins_hash,
                            operation=operation)
        cache_file = self.differential_cache / f"{key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
        
        self._log_access('set', 'differential', key, hit=True,
                        metadata={'modes': f"{mode_a}_vs_{mode_b}",
                                'operation': operation})
    
    def compute_with_cache(self, compute_func, mode: str, dest_type: str,
                          threshold: int, origins_hash: str, **kwargs):
        """Compute accessibility with automatic caching."""
        cached = self.get_absolute(mode, dest_type, threshold, origins_hash)
        if cached is not None:
            return cached
        
        result = compute_func(mode=mode, dest_type=dest_type, 
                            threshold=threshold, **kwargs)
        self.set_absolute(result, mode, dest_type, threshold, origins_hash)
        return result
    
    def compute_differential(self, value_a: Any, value_b: Any,
                           mode_a: str, mode_b: str, dest_type: str,
                           threshold: int, origins_hash: str,
                           operation: str = 'ratio') -> Any:
        """
        Compute and cache differential between two accessibility results.
        
        operation: 'ratio' (a/b), 'difference' (a-b), or 'log_ratio' (log(a/b))
        """
        cached = self.get_differential(mode_a, mode_b, dest_type, threshold,
                                      origins_hash, operation)
        if cached is not None:
            return cached
        
        if operation == 'ratio':
            result = value_a / value_b
        elif operation == 'difference':
            result = value_a - value_b
        elif operation == 'log_ratio':
            result = pd.Series(value_a / value_b).apply(lambda x: 
                              pd.np.log(x) if x > 0 else pd.np.nan)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        self.set_differential(result, mode_a, mode_b, dest_type, threshold,
                            origins_hash, operation)
        return result


class WorkloadAnalyzer:
    """Analyze cache access patterns to understand workload characteristics."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.logs = self._load_logs()
    
    def _load_logs(self) -> pd.DataFrame:
        """Load all log entries into DataFrame."""
        if not self.log_file.exists():
            return pd.DataFrame()
        
        logs = []
        with open(self.log_file, 'r') as f:
            for line in f:
                logs.append(json.loads(line))
        
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def hit_rate(self, cache_type: str = None) -> float:
        """Calculate cache hit rate."""
        logs = self.logs[self.logs['operation'] == 'get']
        if cache_type:
            logs = logs[logs['cache_type'] == cache_type]
        
        if len(logs) == 0:
            return 0.0
        return logs['hit'].sum() / len(logs)
    
    def most_common_queries(self, n: int = 10) -> pd.DataFrame:
        """Identify most frequently accessed cache keys."""
        logs = self.logs[self.logs['operation'] == 'get']
        return logs.groupby('key').size().sort_values(ascending=False).head(n)
    
    def mode_comparison_patterns(self) -> pd.DataFrame:
        """Analyze which mode comparisons are most common."""
        diff_logs = self.logs[
            (self.logs['cache_type'] == 'differential') & 
            (self.logs['operation'] == 'get')
        ]
        
        if len(diff_logs) == 0:
            return pd.DataFrame()
        
        comparisons = diff_logs['metadata'].apply(
            lambda x: x.get('modes', 'unknown')
        )
        return comparisons.value_counts()
    
    def temporal_access_pattern(self) -> pd.DataFrame:
        """Show access patterns over time."""
        return self.logs.groupby(
            self.logs['timestamp'].dt.floor('H')
        ).size()
    
    def differential_vs_absolute_usage(self) -> Dict[str, int]:
        """Compare how often differential vs absolute caches are used."""
        return self.logs[self.logs['operation'] == 'get'].groupby(
            'cache_type'
        ).size().to_dict()
    
    def generate_report(self) -> str:
        """Generate workload analysis report."""
        report = []
        report.append("=" * 60)
        report.append("GRANITE Cache Workload Analysis Report")
        report.append("=" * 60)
        report.append(f"\nTotal cache accesses: {len(self.logs)}")
        report.append(f"Absolute cache hit rate: {self.hit_rate('absolute'):.2%}")
        report.append(f"Differential cache hit rate: {self.hit_rate('differential'):.2%}")
        
        usage = self.differential_vs_absolute_usage()
        report.append(f"\nCache type usage:")
        for cache_type, count in usage.items():
            report.append(f"  {cache_type}: {count} accesses")
        
        report.append(f"\nMost common mode comparisons:")
        for mode_pair, count in self.mode_comparison_patterns().head(5).items():
            report.append(f"  {mode_pair}: {count} times")
        
        return "\n".join(report)