#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Base Predictor - Base class for all prediction strategies
Provides unified interface for data preparation and result generation
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """
    Base class for all predictors
    
    All predictors should inherit from this class and implement:
    1. prepare_data() - Prepare necessary data (e.g., generate charts, calculate indicators)
    2. analyze() - Core analysis logic
    3. generate_output() - Generate standardized output
    """
    
    def __init__(self, symbol: str, name: str = "default"):
        """
        Initialize predictor
        
        Args:
            symbol: Trading pair symbol
            name: Strategy name
        """
        self.symbol = symbol
        self.name = name
    
    @abstractmethod
    def prepare_data(self, df, df_display, ma_dict, ma_dict_display, stats, args) -> Tuple[Optional[str], Optional[bytes]]:
        """
        Prepare necessary data for analysis
        
        Args:
            df: Full dataframe
            df_display: Display dataframe (limited rows)
            ma_dict: Full MA dictionary
            ma_dict_display: Display MA dictionary
            stats: Statistics dictionary
            args: Command line arguments
            
        Returns:
            Tuple of (save_path, image_bytes)
        """
        pass
    
    @abstractmethod
    def analyze(self, stats: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Core analysis logic
        
        Args:
            stats: Statistics dictionary containing market data
            **kwargs: Additional analysis parameters
            
        Returns:
            Analysis result dictionary
        """
        pass
    
    @abstractmethod
    def generate_output(self, result: Dict[str, Any], stats: Dict[str, Any], args) -> Dict[str, Any]:
        """
        Generate standardized output
        
        Args:
            result: Analysis result from analyze()
            stats: Statistics dictionary
            args: Command line arguments
            
        Returns:
            Standardized output dictionary with fields:
            - symbol: Trading symbol
            - position: Position recommendation
            - confidence: Confidence level
            - current_price: Current price
            - reasoning: Decision reasoning
            - analysis_type: Type of analysis
        """
        pass
    
    def run(self, df, df_display, ma_dict, ma_dict_display, stats, args) -> Dict[str, Any]:
        """
        Execute complete prediction pipeline
        
        Args:
            df: Full dataframe
            df_display: Display dataframe
            ma_dict: Full MA dictionary
            ma_dict_display: Display MA dictionary
            stats: Statistics dictionary
            args: Command line arguments
            
        Returns:
            Standardized prediction result
        """
        # Step 1: Prepare data
        save_path, image_bytes = self.prepare_data(
            df, df_display, ma_dict, ma_dict_display, stats, args
        )
        
        # Step 2: Analyze
        analysis_result = self.analyze(
            stats=stats,
            save_path=save_path,
            image_bytes=image_bytes,
            args=args
        )
        
        # Step 3: Generate output
        output = self.generate_output(analysis_result, stats, args)
        
        return output, save_path
