#!/usr/bin/env python
#-*- coding:utf-8 -*-


from binance.fut.usdm import USDM
from binance.auth.utils import load_api_keys


class Trader:
    
    def __init__(self, account: str = 'li'):
        api_key, private_key = load_api_keys(account)
        self.cli = USDM(api_key=api_key, private_key=private_key)
    
    def trade(self, advisor):
        symbol = advisor['symbol']
        curpos = self.cli.account(symbol=symbol)['positionAmt']
        target = advisor['position']
        volume = target - curpos
        if volume == 0:
            print("当前仓位已达目标，无需调整")
            return
        side = 'BUY' if volume > 0 else 'SELL'
        quantity = abs(volume)
        price = advisor['current_price']
        order = dict(
            symbol=symbol, side=side, quantity=quantity,
            type='LIMIT', timeInForce='GTC', price=price)
        self.cli.new_order(**order)