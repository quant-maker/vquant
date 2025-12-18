#!/usr/bin/env python
#-*- coding:utf-8 -*-


import logging
from binance.fut.usdm import USDM
from binance.auth.utils import load_api_keys
from strategy.common.utils import round_at, lot_round_at
from strategy.common.utils import cancel_all, round_it


logger = logging.getLogger(__name__)


class Trader:
    """Trade Executor - Execute trades based on AI analysis results"""
    
    def __init__(self, init_pos, account: str = 'li'):
        """
        Initialize trader
        Args:
            account: Trading account name
        """
        logger.debug(f"Initializing trader, account: {account}")
        api_key, private_key = load_api_keys(account)
        self.cli = USDM(api_key=api_key, private_key=private_key)
        self.account = account
        self.init_pos = init_pos

    def trade(self, advisor, args):
        """
        Execute trade
        Args:
            advisor: AI analysis result dict, must contain symbol, position, price
        """
        symbol = advisor['symbol']
        target = advisor['position']
        price = round_it(advisor['current_price'], round_at(symbol))
        
        logger.info(f"Checking position for {symbol}...")
        logger.debug(f"Target position: {target}, current price: {price}")
        
        # Get current position
        posinfo = self.cli.position(symbol=symbol)
        curpos = float(posinfo[0]['positionAmt']) if posinfo else 0
        logger.info(f"Current position: {curpos}")
        
        # Calculate adjustment volume
        # target = args.volume if target > args.threshold else -args.volume if target < -args.threshold else 0
        target = (args.volume * target) # here target is weighted position ratio and volume is max position size
        volume = target - curpos + self.init_pos
        quantity = round_it(abs(volume), lot_round_at(symbol))
        if float(quantity) == 0:  # Set a tolerance value
            logger.info("Current position already at target, no adjustment needed")
            return
        # Prepare order
        side = 'BUY' if volume > 0 else 'SELL'
        logger.info(f"Preparing order: {side} {quantity} {symbol} @ {price}")
        order = dict(
            symbol=symbol, side=side, quantity=quantity,
            type='LIMIT', timeInForce='GTC', price=price)
        try:
            logger.info(f"Send {order=}")
            result = self.cli.new_order(**order)
            logger.info(f"Order successful: OrderID={result.get('orderId', 'N/A')}")
            logger.debug(f"Order details: {result}")
            return result
        except Exception as e:
            logger.error(f"Order failed: {e}", exc_info=True)
            raise
