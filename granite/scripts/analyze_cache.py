"""
Analyze GRANITE cache usage patterns
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from granite.cache import WorkloadAnalyzer

def main():
    cache_log = Path('./granite_cache/access_log.jsonl')
    
    if not cache_log.exists():
        print("No cache log found. Run GRANITE workflows first to collect data.")
        return
    
    print("\n" + "="*60)
    print("GRANITE Cache Workload Analysis")
    print("="*60 + "\n")
    
    analyzer = WorkloadAnalyzer(cache_log)
    print(analyzer.generate_report())
    
    # Detailed mode comparison analysis
    print("\n" + "="*60)
    print("Mode Comparison Patterns")
    print("="*60)
    comparisons = analyzer.mode_comparison_patterns()
    if len(comparisons) > 0:
        print(comparisons.head(10))
    else:
        print("No mode comparisons cached yet")

if __name__ == '__main__':
    main()