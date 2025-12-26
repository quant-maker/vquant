#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Analysis module - Contains all prediction and analysis strategies
"""

from .base import BasePredictor
from .quant import QuantPredictor
from .wave import WaveTrader
from .martin import MartinTrader
from .kelly import KellyTrader
from .advisor import PositionAdvisor

__all__ = [
    'BasePredictor',
    'QuantPredictor',
    'WaveTrader',
    'MartinTrader',
    'KellyTrader',
    'PositionAdvisor',
]
