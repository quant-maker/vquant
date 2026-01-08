#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Kronos Prediction Scraper - Scrape prediction results from Kronos official website
Website: https://shiyu-coder.github.io/Kronos-demos/
"""

import logging
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class KronosScraper:
    """
    Scraper for Kronos official prediction results
    
    Features:
    1. Scrape latest BTC/USDT prediction from official website
    2. Parse prediction data (price, trend, confidence)
    3. Check update timestamp to avoid trading on stale data
    4. Cache results to reduce request frequency
    """
    
    def __init__(self, url: str = "https://shiyu-coder.github.io/Kronos-demos/", 
                 max_staleness_hours: int = 24,
                 fetch_timeout_minutes: int = 60,
                 max_retries: int = 3):
        """
        Initialize scraper
        
        Args:
            url: Kronos official demo URL
            max_staleness_hours: Maximum allowed data staleness in hours (default: 24)
                                If data is older than this, trading will be blocked
            fetch_timeout_minutes: Maximum minutes to keep trying before giving up (default: 60)
                                  Note: Kronos updates hourly, so 60min timeout recommended
            max_retries: Maximum number of retry attempts per fetch (default: 3)
        """
        self.url = url
        self.max_staleness_hours = max_staleness_hours
        self.fetch_timeout_minutes = fetch_timeout_minutes
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self._cache = None
        self._cache_time = None
        self._cache_ttl = 300  # Cache TTL: 5 minutes
        
        # Failure tracking for timeout mechanism
        self._first_failure_time = None
        self._consecutive_failures = 0
        self._last_successful_fetch = None
    
    def fetch_prediction(self, symbol: str = "BTCUSDT", use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch latest prediction for specified symbol with retry and timeout mechanism
        
        Args:
            symbol: Trading symbol (default: BTCUSDT)
            use_cache: Whether to use cached data
            
        Returns:
            Prediction data dictionary:
            {
                'symbol': str,           # Trading symbol
                'predicted_price': float,  # Predicted price
                'current_price': float,    # Current price at prediction time
                'trend': str,              # 'up', 'down', or 'neutral'
                'confidence': float,       # Confidence score (0-1)
                'update_time': datetime,   # Prediction update time
                'is_stale': bool,          # Whether data is stale
                'staleness_hours': float,  # Hours since last update
                'raw_data': dict,          # Original scraped data
                'fetch_status': str,       # 'success', 'temporary_failure', 'timeout_exceeded'
                'consecutive_failures': int,  # Number of consecutive fetch failures
                'time_since_first_failure': float  # Minutes since first failure
            }
            
            Returns cached data on temporary failure, None only after timeout exceeded
        """
        # Check cache first
        if use_cache and self._cache and self._cache_time:
            if (datetime.now() - self._cache_time).total_seconds() < self._cache_ttl:
                logger.info(f"Using cached prediction (age: {(datetime.now() - self._cache_time).total_seconds():.0f}s)")
                return self._cache
        
        logger.info(f"Fetching prediction from {self.url}...")
        
        # Try to fetch with retries
        prediction = None
        for attempt in range(self.max_retries):
            try:
                # Fetch webpage
                response = self.session.get(self.url, timeout=10)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract prediction data
                prediction = self._parse_html(soup, symbol)
                
                if prediction:
                    # Success! Reset failure tracking
                    self._reset_failure_tracking()
                    self._last_successful_fetch = datetime.now()
                    
                    # Check data freshness
                    update_time = prediction.get('update_time')
                    if update_time:
                        staleness_hours = (datetime.now() - update_time).total_seconds() / 3600
                        prediction['staleness_hours'] = staleness_hours
                        prediction['is_stale'] = staleness_hours > self.max_staleness_hours
                        
                        if prediction['is_stale']:
                            logger.warning(
                                f"⚠️ Prediction data is stale! "
                                f"Last update: {update_time.strftime('%Y-%m-%d %H:%M:%S')} "
                                f"({staleness_hours:.1f} hours ago)"
                            )
                            logger.warning(
                                f"⚠️ Trading is NOT recommended - "
                                f"Official website may have stopped updating!"
                            )
                    else:
                        prediction['staleness_hours'] = None
                        prediction['is_stale'] = True
                        logger.warning("⚠️ Cannot determine data freshness - no timestamp found")
                    
                    # Add fetch status
                    prediction['fetch_status'] = 'success'
                    prediction['consecutive_failures'] = 0
                    prediction['time_since_first_failure'] = 0
                    
                    # Cache result
                    self._cache = prediction
                    self._cache_time = datetime.now()
                    
                    logger.info(f"✓ Successfully fetched prediction for {symbol}")
                    return prediction
                else:
                    logger.warning(f"Failed to parse prediction data for {symbol} (attempt {attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        import time
                        time.sleep(2)  # Wait 2 seconds before retry
                    
            except requests.Timeout:
                logger.warning(f"Request timeout: {self.url} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2)
            except requests.RequestException as e:
                logger.warning(f"Request failed: {e} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Unexpected error while scraping: {e} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2)
        
        # All retries failed - handle failure
        return self._handle_fetch_failure(symbol)
    
    def _parse_html(self, soup: BeautifulSoup, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Parse HTML to extract prediction data
        
        This method needs to be customized based on the actual website structure.
        Below are several common parsing strategies.
        
        Args:
            soup: BeautifulSoup object
            symbol: Trading symbol
            
        Returns:
            Parsed prediction data, or None if parsing fails
        """
        try:
            # Strategy 0: Parse Kronos-specific HTML structure
            kronos_data = self._extract_from_kronos_page(soup, symbol)
            if kronos_data:
                return kronos_data
            
            # Strategy 1: Try to find JSON data in <script> tags
            # Many modern websites embed data in JavaScript
            json_data = self._extract_json_from_script(soup)
            if json_data:
                return self._parse_json_data(json_data, symbol)
            
            # Strategy 2: Try to parse structured HTML tables/divs
            table_data = self._extract_from_table(soup, symbol)
            if table_data:
                return table_data
            
            # Strategy 3: Try to extract from meta tags or specific elements
            meta_data = self._extract_from_meta(soup, symbol)
            if meta_data:
                return meta_data
            
            # If all strategies fail, log the HTML for debugging
            logger.debug(f"HTML content (first 500 chars): {str(soup)[:500]}")
            logger.error("Failed to parse prediction data - website structure may have changed")
            return None
            
        except Exception as e:
            logger.exception(f"Error parsing HTML: {e}")
            return None
    
    def _extract_from_kronos_page(self, soup: BeautifulSoup, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Extract prediction data from Kronos official demo page
        
        The page structure:
        - <p class="metric-value" id="upside-prob">90.0%</p>
        - <p class="metric-value" id="vol-amp-prob">33.3%</p>
        - <span>Last Updated (UTC): <strong id="update-time">2026-01-08 04:00:44</strong></span>
        """
        try:
            # Extract upside probability
            upside_elem = soup.find('p', id='upside-prob')
            if not upside_elem:
                logger.debug("Could not find 'upside-prob' element")
                return None
            
            upside_prob_text = upside_elem.get_text(strip=True)
            upside_prob = float(upside_prob_text.replace('%', '')) / 100.0
            
            # Extract volatility amplification probability (optional)
            vol_elem = soup.find('p', id='vol-amp-prob')
            vol_amp_prob = None
            if vol_elem:
                vol_prob_text = vol_elem.get_text(strip=True)
                vol_amp_prob = float(vol_prob_text.replace('%', '')) / 100.0
            
            # Extract update time
            time_elem = soup.find('strong', id='update-time')
            update_time = None
            time_str = None
            if time_elem:
                time_str = time_elem.get_text(strip=True)
                update_time = self._parse_timestamp(time_str)
            
            # Construct prediction
            # Upside probability > 50% means upward trend
            if upside_prob > 0.5:
                trend = 'up'
                confidence = upside_prob
            elif upside_prob < 0.5:
                trend = 'down'
                confidence = 1.0 - upside_prob
            else:
                trend = 'neutral'
                confidence = 0.5
            
            prediction = {
                'symbol': symbol,
                'predicted_price': None,  # Not provided on the page
                'current_price': None,    # Not provided on the page
                'trend': trend,
                'confidence': confidence,
                'update_time': update_time or datetime.now(),
                'raw_data': {
                    'upside_probability': upside_prob,
                    'volatility_amplification_probability': vol_amp_prob,
                    'update_time': time_str if time_elem else None
                }
            }
            
            logger.info(f"Successfully parsed Kronos prediction: trend={trend}, confidence={confidence:.2%}, upside_prob={upside_prob:.2%}")
            return prediction
            
        except Exception as e:
            logger.debug(f"Failed to parse Kronos-specific structure: {e}")
            return None
    
    def _extract_json_from_script(self, soup: BeautifulSoup) -> Optional[dict]:
        """Extract JSON data from <script> tags"""
        script_tags = soup.find_all('script')
        
        for script in script_tags:
            if script.string:
                # Look for JSON patterns
                # Common patterns: var data = {...}, window.data = {...}, etc.
                json_patterns = [
                    r'var\s+\w+\s*=\s*(\{.*?\});',
                    r'window\.\w+\s*=\s*(\{.*?\});',
                    r'const\s+\w+\s*=\s*(\{.*?\});',
                    r'(\{["\'].*?:.*?\})',  # Generic JSON object
                ]
                
                for pattern in json_patterns:
                    matches = re.findall(pattern, script.string, re.DOTALL)
                    for match in matches:
                        try:
                            data = json.loads(match)
                            if self._is_valid_prediction_data(data):
                                return data
                        except json.JSONDecodeError:
                            continue
        
        return None
    
    def _extract_from_table(self, soup: BeautifulSoup, symbol: str) -> Optional[Dict[str, Any]]:
        """Extract data from HTML tables"""
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            data = {}
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)
                    data[key] = value
            
            # Try to construct prediction from table data
            if 'price' in data or 'prediction' in data:
                return self._construct_prediction_from_raw(data, symbol)
        
        return None
    
    def _extract_from_meta(self, soup: BeautifulSoup, symbol: str) -> Optional[Dict[str, Any]]:
        """Extract data from meta tags or specific divs/spans"""
        # Look for specific elements by class/id
        prediction_elements = soup.find_all(['div', 'span'], class_=re.compile(r'predict|price|forecast', re.I))
        
        data = {}
        for elem in prediction_elements:
            text = elem.get_text(strip=True)
            # Extract numbers that might be prices
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                data[elem.get('class', ['unknown'])[0]] = numbers
        
        if data:
            return self._construct_prediction_from_raw(data, symbol)
        
        return None
    
    def _is_valid_prediction_data(self, data: dict) -> bool:
        """Check if JSON data contains valid prediction information"""
        # Check for common keys that might indicate prediction data
        prediction_keys = ['price', 'prediction', 'forecast', 'btc', 'usdt', 'trend']
        return any(key in str(data).lower() for key in prediction_keys)
    
    def _construct_prediction_from_raw(self, raw_data: dict, symbol: str) -> Dict[str, Any]:
        """
        Construct standardized prediction dictionary from raw data
        
        This is a template that should be adjusted based on actual data format
        """
        # Try to extract key values
        predicted_price = None
        current_price = None
        confidence = 0.5  # Default confidence
        update_time = None
        
        # Extract predicted price
        for key in ['predicted_price', 'prediction', 'forecast', 'target_price']:
            if key in raw_data:
                try:
                    predicted_price = float(raw_data[key])
                    break
                except (ValueError, TypeError):
                    continue
        
        # Extract current price
        for key in ['current_price', 'price', 'spot_price']:
            if key in raw_data:
                try:
                    current_price = float(raw_data[key])
                    break
                except (ValueError, TypeError):
                    continue
        
        # Extract confidence
        for key in ['confidence', 'score', 'probability']:
            if key in raw_data:
                try:
                    confidence = float(raw_data[key])
                    break
                except (ValueError, TypeError):
                    continue
        
        # Extract timestamp
        for key in ['timestamp', 'time', 'update_time', 'date']:
            if key in raw_data:
                try:
                    # Try to parse various time formats
                    time_str = str(raw_data[key])
                    update_time = self._parse_timestamp(time_str)
                    break
                except:
                    continue
        
        # Determine trend
        if predicted_price and current_price:
            if predicted_price > current_price * 1.01:
                trend = 'up'
            elif predicted_price < current_price * 0.99:
                trend = 'down'
            else:
                trend = 'neutral'
        else:
            trend = 'unknown'
        
        return {
            'symbol': symbol,
            'predicted_price': predicted_price,
            'current_price': current_price,
            'trend': trend,
            'confidence': confidence,
            'update_time': update_time or datetime.now(),  # Use current time if not found
            'raw_data': raw_data
        }
    
    def _parse_timestamp(self, time_str: str) -> Optional[datetime]:
        """Parse timestamp string into datetime object"""
        # Try common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d %H:%M',
            '%d-%m-%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        # Try Unix timestamp
        try:
            timestamp = float(time_str)
            # Check if it's in milliseconds
            if timestamp > 10**12:
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp)
        except:
            pass
        
        logger.warning(f"Failed to parse timestamp: {time_str}")
        return None
    
    def _reset_failure_tracking(self):
        """Reset failure tracking after successful fetch"""
        self._first_failure_time = None
        self._consecutive_failures = 0
    
    def _handle_fetch_failure(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Handle fetch failure with timeout mechanism
        
        Strategy:
        1. Track first failure time and consecutive failures
        2. If within timeout window: return cached data (keep position)
        3. If timeout exceeded: return None (clear position)
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Cached prediction with failure status, or None if timeout exceeded
        """
        # Update failure tracking
        self._consecutive_failures += 1
        if self._first_failure_time is None:
            self._first_failure_time = datetime.now()
            logger.warning(f"⚠️ First fetch failure detected at {self._first_failure_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate time since first failure
        time_since_first_failure = (datetime.now() - self._first_failure_time).total_seconds() / 60  # in minutes
        
        logger.warning(
            f"⚠️ Fetch failed {self._consecutive_failures} times. "
            f"Time since first failure: {time_since_first_failure:.1f} minutes "
            f"(timeout: {self.fetch_timeout_minutes} minutes)"
        )
        
        # Check if timeout exceeded
        if time_since_first_failure > self.fetch_timeout_minutes:
            logger.error(
                f"❌ TIMEOUT EXCEEDED! Failed to fetch data for {self.fetch_timeout_minutes} minutes. "
                f"Consecutive failures: {self._consecutive_failures}. "
                f"Clearing position and stopping trading!"
            )
            return None
        
        # Within timeout window - return cached data if available
        if self._cache:
            logger.warning(
                f"⚠️ Temporary fetch failure - returning cached data to maintain position. "
                f"Will timeout in {self.fetch_timeout_minutes - time_since_first_failure:.1f} minutes."
            )
            
            # Clone cache and update status
            cached_prediction = self._cache.copy()
            cached_prediction['fetch_status'] = 'temporary_failure'
            cached_prediction['consecutive_failures'] = self._consecutive_failures
            cached_prediction['time_since_first_failure'] = time_since_first_failure
            
            return cached_prediction
        else:
            logger.error(
                f"❌ No cached data available and fetch failed. "
                f"Cannot maintain position safely. Clearing position!"
            )
            return None
    
    def is_safe_to_trade(self, prediction: Dict[str, Any]) -> bool:
        """
        Check if it's safe to trade based on prediction data freshness
        
        Args:
            prediction: Prediction data dictionary
            
        Returns:
            True if data is fresh enough for trading, False otherwise
        """
        if not prediction:
            logger.warning("No prediction data - trading NOT safe")
            return False
        
        if prediction.get('is_stale', True):
            staleness = prediction.get('staleness_hours', 'unknown')
            logger.warning(
                f"⚠️ Trading NOT safe - data is {staleness} hours old "
                f"(max allowed: {self.max_staleness_hours} hours)"
            )
            return False
        
        logger.info("✓ Data is fresh - safe to trade")
        return True
    
    def get_position_signal(self, prediction: Dict[str, Any]) -> float:
        """
        Convert Kronos prediction to position signal
        
        Args:
            prediction: Prediction data dictionary
            
        Returns:
            Position signal from -1.0 (full short) to 1.0 (full long)
        """
        if not prediction:
            return 0.0
        
        trend = prediction.get('trend', 'neutral')
        confidence = prediction.get('confidence', 0.5)
        
        # Convert trend to position
        if trend == 'up':
            position = confidence
        elif trend == 'down':
            position = -confidence
        else:
            position = 0.0
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, position))


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s'
    )
    
    scraper = KronosScraper(max_staleness_hours=24)
    
    print("\n=== Testing Kronos Scraper ===\n")
    
    prediction = scraper.fetch_prediction("BTCUSDT")
    
    if prediction:
        print("✓ Scraping successful!")
        print(f"\nSymbol: {prediction['symbol']}")
        print(f"Predicted Price: {prediction.get('predicted_price', 'N/A')}")
        print(f"Current Price: {prediction.get('current_price', 'N/A')}")
        print(f"Trend: {prediction['trend']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        
        if prediction.get('update_time'):
            print(f"Update Time: {prediction['update_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Staleness: {prediction.get('staleness_hours', 0):.1f} hours")
        
        print(f"\nIs Stale: {prediction.get('is_stale', 'Unknown')}")
        print(f"Safe to Trade: {scraper.is_safe_to_trade(prediction)}")
        print(f"Position Signal: {scraper.get_position_signal(prediction):.2f}")
        
        print(f"\nRaw Data: {prediction.get('raw_data', {})}")
    else:
        print("✗ Scraping failed")
        print("\nNote: This scraper needs to be customized for the actual Kronos website structure.")
        print("Please inspect the website HTML and adjust the parsing logic accordingly.")
