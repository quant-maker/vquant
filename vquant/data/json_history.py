#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
JSON History Manager - Manage historical JSON results for position fallback
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class JSONHistoryManager:
    """
    Manage JSON history for position fallback mechanism
    
    Features:
    1. Find most recent successful JSON result
    2. Load prediction data from JSON
    3. Check if JSON is within timeout window
    4. Clean up old JSON files
    """
    
    def __init__(self, 
                 strategy_name: str,
                 history_dir: str = "charts",
                 max_age_minutes: int = 30):
        """
        Initialize JSON history manager
        
        Args:
            strategy_name: Strategy name (used to filter JSON files)
            history_dir: Directory containing JSON history files
            max_age_minutes: Maximum age of JSON to use (in minutes)
        """
        self.strategy_name = strategy_name
        self.history_dir = Path(history_dir)
        self.max_age_minutes = max_age_minutes
        
        # Ensure directory exists
        self.history_dir.mkdir(parents=True, exist_ok=True)
    
    def find_recent_successful_json(self) -> Optional[Path]:
        """
        Find the most recent successful JSON result
        
        Returns:
            Path to most recent successful JSON, or None if not found
        """
        pattern = f"{self.strategy_name}_*.json"
        json_files = list(self.history_dir.glob(pattern))
        
        if not json_files:
            logger.warning(f"No JSON history found for strategy: {self.strategy_name}")
            return None
        
        # Sort by modification time (newest first)
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Find first successful result
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if it's a successful fetch
                fetch_status = data.get('fetch_status', 'unknown')
                position = data.get('position', 0)
                
                if fetch_status == 'success' or (fetch_status == 'unknown' and position != 0):
                    logger.info(f"Found recent successful JSON: {json_file.name}")
                    return json_file
                    
            except Exception as e:
                logger.warning(f"Failed to read JSON {json_file.name}: {e}")
                continue
        
        logger.warning(f"No successful JSON found in recent history")
        return None
    
    def get_json_age_minutes(self, json_path: Path) -> float:
        """
        Get age of JSON file in minutes
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Age in minutes
        """
        mtime = datetime.fromtimestamp(json_path.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds() / 60
        return age
    
    def is_json_within_timeout(self, json_path: Path) -> bool:
        """
        Check if JSON is within timeout window
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            True if within timeout, False otherwise
        """
        age_minutes = self.get_json_age_minutes(json_path)
        is_valid = age_minutes <= self.max_age_minutes
        
        if not is_valid:
            logger.warning(
                f"JSON age ({age_minutes:.1f} min) exceeds timeout ({self.max_age_minutes} min)"
            )
        
        return is_valid
    
    def load_prediction_from_json(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load prediction data from JSON file
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Prediction dictionary, or None if failed
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract Kronos-specific data
            kronos_data = data.get('kronos_data', {})
            
            # Reconstruct prediction format
            prediction = {
                'symbol': data.get('symbol'),
                'trend': kronos_data.get('trend'),
                'confidence': kronos_data.get('kronos_confidence'),
                'predicted_price': kronos_data.get('predicted_price'),
                'current_price': data.get('current_price'),
                'update_time': self._parse_datetime(kronos_data.get('update_time')),
                'staleness_hours': kronos_data.get('staleness_hours'),
                'is_stale': kronos_data.get('is_stale', True),
                'raw_data': {},
                # Add fallback metadata
                'fetch_status': 'json_fallback',
                'json_age_minutes': self.get_json_age_minutes(json_path),
                'json_source': str(json_path.name)
            }
            
            logger.info(
                f"Loaded prediction from JSON: {json_path.name} "
                f"(age: {prediction['json_age_minutes']:.1f} min)"
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to load prediction from JSON {json_path.name}: {e}")
            return None
    
    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string from JSON"""
        if not dt_str:
            return None
        
        try:
            # ISO format: 2026-01-08T04:00:44
            return datetime.fromisoformat(dt_str)
        except Exception as e:
            logger.warning(f"Failed to parse datetime: {dt_str} - {e}")
            return None
    
    def get_fallback_prediction(self) -> Optional[Dict[str, Any]]:
        """
        Get fallback prediction from JSON history
        
        This is the main method to use when fetch fails.
        
        Returns:
            Prediction data if available and within timeout, None otherwise
        """
        # Find recent successful JSON
        json_path = self.find_recent_successful_json()
        
        if not json_path:
            logger.error("No JSON history available for fallback")
            return None
        
        # Check if within timeout window
        if not self.is_json_within_timeout(json_path):
            logger.error(f"Most recent JSON is too old (> {self.max_age_minutes} min)")
            return None
        
        # Load prediction
        prediction = self.load_prediction_from_json(json_path)
        
        if prediction:
            logger.warning(
                f"⚠️ Using JSON fallback to maintain position. "
                f"Source: {prediction['json_source']}, "
                f"Age: {prediction['json_age_minutes']:.1f} min"
            )
        
        return prediction
    
    def cleanup_old_files(self, days: int = 7) -> int:
        """
        Clean up old JSON files
        
        Args:
            days: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        pattern = f"{self.strategy_name}_*.json"
        
        deleted_count = 0
        for json_file in self.history_dir.glob(pattern):
            mtime = datetime.fromtimestamp(json_file.stat().st_mtime)
            
            if mtime < cutoff_time:
                try:
                    json_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old JSON: {json_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {json_file.name}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old JSON files (older than {days} days)")
        
        return deleted_count
    
    def get_history_stats(self) -> Dict[str, Any]:
        """
        Get statistics about JSON history
        
        Returns:
            Statistics dictionary
        """
        pattern = f"{self.strategy_name}_*.json"
        json_files = list(self.history_dir.glob(pattern))
        
        if not json_files:
            return {
                'total_files': 0,
                'success_count': 0,
                'failure_count': 0,
                'oldest_file': None,
                'newest_file': None
            }
        
        success_count = 0
        failure_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                fetch_status = data.get('fetch_status', 'unknown')
                if fetch_status == 'success':
                    success_count += 1
                elif fetch_status in ('temporary_failure', 'timeout_exceeded'):
                    failure_count += 1
            except:
                pass
        
        oldest_file = min(json_files, key=lambda f: f.stat().st_mtime)
        newest_file = max(json_files, key=lambda f: f.stat().st_mtime)
        
        return {
            'total_files': len(json_files),
            'success_count': success_count,
            'failure_count': failure_count,
            'oldest_file': oldest_file.name,
            'newest_file': newest_file.name,
            'oldest_age_hours': (datetime.now() - datetime.fromtimestamp(oldest_file.stat().st_mtime)).total_seconds() / 3600,
            'newest_age_minutes': (datetime.now() - datetime.fromtimestamp(newest_file.stat().st_mtime)).total_seconds() / 60
        }


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s'
    )
    
    print("\n=== Testing JSON History Manager ===\n")
    
    # Create manager
    manager = JSONHistoryManager(
        strategy_name="test",
        history_dir="charts",
        max_age_minutes=30
    )
    
    # Get statistics
    print("1. Getting history statistics...")
    stats = manager.get_history_stats()
    print(f"   Total files: {stats['total_files']}")
    print(f"   Success count: {stats['success_count']}")
    print(f"   Failure count: {stats['failure_count']}")
    if stats['total_files'] > 0:
        print(f"   Newest file: {stats['newest_file']} ({stats['newest_age_minutes']:.1f} min ago)")
        print(f"   Oldest file: {stats['oldest_file']} ({stats['oldest_age_hours']:.1f} hours ago)")
    
    # Test fallback
    print("\n2. Testing fallback mechanism...")
    fallback = manager.get_fallback_prediction()
    if fallback:
        print(f"   ✓ Fallback available")
        print(f"   Source: {fallback['json_source']}")
        print(f"   Age: {fallback['json_age_minutes']:.1f} minutes")
        print(f"   Trend: {fallback['trend']}")
        print(f"   Confidence: {fallback['confidence']}")
    else:
        print("   ✗ No fallback available")
    
    # Test cleanup (dry run)
    print("\n3. Checking for old files...")
    old_count = len([f for f in manager.history_dir.glob(f"{manager.strategy_name}_*.json")
                     if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days > 7])
    print(f"   Files older than 7 days: {old_count}")
    
    print("\n=== Test Complete ===\n")
