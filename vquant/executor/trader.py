#!/usr/bin/env python
#-*- coding:utf-8 -*-


import logging
from binance.fut.usdm import USDM
from binance.auth.utils import load_api_keys
from strategy.common.utils import round_at, lot_round_at
from strategy.common.utils import cancel_all, round_it
from vquant.executor.position import PositionManager


logger = logging.getLogger(__name__)


class Trader:
    """Trade Executor - Execute trades based on AI analysis results"""
    
    def __init__(self, name, account: str = 'li'):
        """
        Initialize trader
        Args:
            account: Trading account name
            name: Strategy name (unique identifier)
        """
        logger.debug(f"Initializing trader, account: {account}, name: {name}")
        api_key, private_key = load_api_keys(account)
        self.cli = USDM(api_key=api_key, private_key=private_key)
        self.account = account
        self.name = name
        
        # Initialize position manager
        self.pm = PositionManager(self.cli, name)

    def trade(self, advisor, args):
        """
        Execute trade with single order strategy:
        1. Check and cancel previous unfilled order
        2. Calculate target position based on current holdings
        3. Place new order if needed
        
        Args:
            advisor: AI analysis result dict, must contain symbol, position, price
        """
        symbol = advisor['symbol']
        target = advisor['position']
        fprice = advisor['current_price']
        price = round_it(fprice, round_at(symbol))
        # Step 1: Get current position (will check and handle previous order automatically)
        curpos = self.pm.get_current_position()
        # Step 2: Calculate target volume
        # target = args.volume if target > args.threshold else -args.volume if target < -args.threshold else 0
        target_pos = (args.volume * target)  # target is weighted position ratio [-1, 1]
        volume = target_pos - curpos # volume to trade
        is_open = (curpos == 0) or (curpos * volume > 0)
        quantity = round_it(abs(volume), lot_round_at(symbol))
        if float(quantity) == 0:
            logger.info("Position already at target, no adjustment needed")
            return
        if is_open and (volume * fprice < 5.2):
            logger.warning(f"Order value too small to open new position: {volume} * {fprice} < 5.2")
            return
        # Step 3: Place new order
        side = 'BUY' if volume > 0 else 'SELL'
        order = dict(
            symbol=symbol, side=side, quantity=quantity,
            type='LIMIT', timeInForce='GTC', price=price)
        if not is_open:
            order['reduceOnly'] = True
        try:
            logger.info(f"Sending order: {order}")
            result = self.cli.new_order(**order)
            order_id = result.get('orderId')
            logger.info(f"Order placed successfully: OrderID={order_id}")
            logger.debug(f"Order details: {result}")
            # Save order state with current position (before this order)
            self.pm.save_new_order(symbol, order_id, side, float(quantity), curpos)
            return result
        except Exception as e:
            logger.exception(f"Order failed: {e}", exc_info=True)
            raise e
