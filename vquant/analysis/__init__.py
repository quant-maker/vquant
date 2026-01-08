#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Analysis module - Contains all prediction and analysis strategies
"""

from .base import BasePredictor
from .quant import QuantPredictor
from .martin import MartinTrader
from .kelly import KellyTrader
from .advisor import PositionAdvisor
from .kronos import KronosTrader
from .kronos_quant import KronosQuantTrader

__all__ = [
    'BasePredictor',
    'QuantPredictor',
    'MartinTrader',
    'KellyTrader',
    'PositionAdvisor',
    'KronosTrader',
    'KronosQuantTrader',
]
