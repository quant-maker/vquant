#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Kronos Strategy Monitor - Monitor Kronos strategy health and performance
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
from collections import defaultdict


def analyze_recent_runs(strategy_name: str = "kronos_prod", 
                       charts_dir: str = "charts",
                       hours: int = 24) -> Dict[str, Any]:
    """
    Analyze recent strategy runs
    
    Args:
        strategy_name: Strategy name
        charts_dir: Directory containing JSON results
        hours: Look back this many hours
        
    Returns:
        Analysis statistics
    """
    charts_path = Path(charts_dir)
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    # Find relevant JSON files
    pattern = f"{strategy_name}_*.json"
    json_files = []
    
    for f in charts_path.glob(pattern):
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime > cutoff_time:
            json_files.append((f, mtime))
    
    # Sort by time
    json_files.sort(key=lambda x: x[1])
    
    # Analyze
    stats = {
        'total_runs': len(json_files),
        'success_count': 0,
        'temporary_failure_count': 0,
        'json_fallback_count': 0,
        'timeout_count': 0,
        'position_changes': [],
        'recent_positions': [],
        'status_distribution': defaultdict(int),
        'confidence_distribution': defaultdict(int),
        'time_range': {
            'start': json_files[0][1] if json_files else None,
            'end': json_files[-1][1] if json_files else None
        }
    }
    
    prev_position = None
    
    for json_file, mtime in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            fetch_status = data.get('fetch_status', 'unknown')
            position = data.get('position', 0)
            confidence = data.get('confidence', 'unknown')
            
            # Count status
            stats['status_distribution'][fetch_status] += 1
            stats['confidence_distribution'][confidence] += 1
            
            if fetch_status == 'success':
                stats['success_count'] += 1
            elif fetch_status == 'temporary_failure':
                stats['temporary_failure_count'] += 1
            elif fetch_status == 'json_fallback':
                stats['json_fallback_count'] += 1
            elif fetch_status == 'timeout_exceeded':
                stats['timeout_count'] += 1
            
            # Track position changes
            if prev_position is not None and prev_position != position:
                stats['position_changes'].append({
                    'time': mtime,
                    'from': prev_position,
                    'to': position,
                    'file': json_file.name
                })
            
            prev_position = position
            
            # Keep recent positions (last 10)
            if len(stats['recent_positions']) < 10:
                stats['recent_positions'].append({
                    'time': mtime,
                    'position': position,
                    'status': fetch_status,
                    'confidence': confidence
                })
            
        except Exception as e:
            print(f"Warning: Failed to read {json_file.name}: {e}")
            continue
    
    # Calculate success rate
    if stats['total_runs'] > 0:
        stats['success_rate'] = stats['success_count'] / stats['total_runs'] * 100
        stats['fallback_rate'] = (stats['temporary_failure_count'] + stats['json_fallback_count']) / stats['total_runs'] * 100
    else:
        stats['success_rate'] = 0
        stats['fallback_rate'] = 0
    
    return stats


def print_report(stats: Dict[str, Any], verbose: bool = False):
    """Print analysis report"""
    
    print("\n" + "=" * 70)
    print(f"Kronos Strategy Health Report")
    print("=" * 70)
    
    # Time range
    if stats['time_range']['start']:
        print(f"\nTime Range:")
        print(f"  Start: {stats['time_range']['start'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End:   {stats['time_range']['end'].strftime('%Y-%m-%d %H:%M:%S')}")
        duration = (stats['time_range']['end'] - stats['time_range']['start']).total_seconds() / 3600
        print(f"  Duration: {duration:.1f} hours")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Total Runs: {stats['total_runs']}")
    print(f"  Success Rate: {stats['success_rate']:.1f}%")
    print(f"  Fallback Rate: {stats['fallback_rate']:.1f}%")
    
    # Status distribution
    print(f"\nStatus Distribution:")
    for status, count in sorted(stats['status_distribution'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_runs'] * 100 if stats['total_runs'] > 0 else 0
        emoji = {
            'success': '✅',
            'temporary_failure': '⚠️',
            'json_fallback': '📁',
            'timeout_exceeded': '❌',
            'unknown': '❓'
        }.get(status, '•')
        print(f"  {emoji} {status:20s}: {count:3d} ({pct:5.1f}%)")
    
    # Confidence distribution
    print(f"\nConfidence Distribution:")
    for conf, count in sorted(stats['confidence_distribution'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_runs'] * 100 if stats['total_runs'] > 0 else 0
        print(f"  {conf:10s}: {count:3d} ({pct:5.1f}%)")
    
    # Recent positions
    if stats['recent_positions']:
        print(f"\nRecent Positions (last {len(stats['recent_positions'])}):")
        for item in reversed(stats['recent_positions']):
            status_emoji = {
                'success': '✅',
                'temporary_failure': '⚠️',
                'json_fallback': '📁',
                'timeout_exceeded': '❌',
                'unknown': '❓'
            }.get(item['status'], '•')
            print(f"  {item['time'].strftime('%m-%d %H:%M')} | "
                  f"Position: {item['position']:+.2f} | "
                  f"Status: {status_emoji} {item['status']:15s} | "
                  f"Confidence: {item['confidence']}")
    
    # Position changes
    if stats['position_changes']:
        print(f"\nPosition Changes ({len(stats['position_changes'])}):")
        for change in stats['position_changes'][-5:]:  # Show last 5
            print(f"  {change['time'].strftime('%m-%d %H:%M')} | "
                  f"{change['from']:+.2f} → {change['to']:+.2f} | "
                  f"File: {change['file']}")
    else:
        print(f"\nPosition Changes: None (stable position)")
    
    # Health assessment
    print(f"\n" + "-" * 70)
    print(f"Health Assessment:")
    
    health_issues = []
    
    if stats['success_rate'] < 50:
        health_issues.append(f"  ⚠️  Low success rate ({stats['success_rate']:.1f}%)")
    
    if stats['fallback_rate'] > 50:
        health_issues.append(f"  ⚠️  High fallback rate ({stats['fallback_rate']:.1f}%)")
    
    if stats['timeout_count'] > 0:
        health_issues.append(f"  ❌ Timeout events detected ({stats['timeout_count']})")
    
    if stats['total_runs'] == 0:
        health_issues.append(f"  ❌ No runs detected")
    
    if health_issues:
        print("\n".join(health_issues))
    else:
        print(f"  ✅ All systems healthy")
    
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor Kronos strategy health")
    parser.add_argument('--name', default='kronos_prod', help='Strategy name (default: kronos_prod)')
    parser.add_argument('--charts-dir', default='charts', help='Charts directory (default: charts)')
    parser.add_argument('--hours', type=int, default=24, help='Look back hours (default: 24)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Analyze
    stats = analyze_recent_runs(
        strategy_name=args.name,
        charts_dir=args.charts_dir,
        hours=args.hours
    )
    
    # Print report
    print_report(stats, verbose=args.verbose)
    
    # Exit code based on health
    if stats['timeout_count'] > 0 or stats['success_rate'] < 50:
        return 1  # Unhealthy
    else:
        return 0  # Healthy


if __name__ == "__main__":
    exit(main())
