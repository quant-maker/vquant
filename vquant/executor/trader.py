#!/usr/bin/env python
#-*- coding:utf-8 -*-


import logging
from binance.fut.usdm import USDM
from binance.auth.utils import load_api_keys


logger = logging.getLogger(__name__)


class Trader:
    """Trade Executor - Execute trades based on AI analysis results"""
    
    def __init__(self, account: str = 'li'):
        """
        Initialize trader
        Args:
            account: Trading account name
        """
        logger.debug(f"Initializing trader, account: {account}")
        api_key, private_key = load_api_keys(account)
        self.cli = USDM(api_key=api_key, private_key=private_key)
        self.account = account
    
    def trade(self, advisor):
        """
        Execute trade
        Args:
            advisor: AI analysis result dict, must contain symbol, position, current_price
        """
        symbol = advisor['symbol']
        target = advisor['position']
        current_price = advisor['current_price']
        
        logger.info(f"Checking position for {symbol}...")
        logger.debug(f"Target position: {target}, current price: {current_price}")
        
        # Get current position
        account_info = self.cli.account(symbol=symbol)
        curpos = float(account_info['positionAmt'])
        logger.info(f"Current position: {curpos}")
        
        # Calculate adjustment volume
        volume = target - curpos
        
        if abs(volume) < 0.001:  # Set a tolerance value
            logger.info("Current position already at target, no adjustment needed")
            return
        
        # Prepare order
        side = 'BUY' if volume > 0 else 'SELL'
        quantity = abs(volume)
        
        logger.info(f"Preparing order: {side} {quantity} {symbol} @ {current_price}")
        
        order = dict(
            symbol=symbol, 
            side=side, 
            quantity=quantity,
            type='LIMIT', 
            timeInForce='GTC', 
            price=current_price
        )
        
        try:
            result = self.cli.new_order(**order)
            logger.info(f"Order successful: OrderID={result.get('orderId', 'N/A')}")
            logger.debug(f"Order details: {result}")
            return result
        except Exception as e:
            logger.error(f"Order failed: {e}", exc_info=True)
            raise