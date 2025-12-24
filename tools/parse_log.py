#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
交易日志解析和PNL计算工具
"""


import sys

from decimal import Decimal
from datetime import datetime
from typing import List, Dict


class Trade:
    """交易记录"""

    def __init__(
        self, timestamp: str, position: float, side: str, quantity: float, price: float
    ):
        self.timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        self.position = position
        self.side = side
        self.quantity = quantity
        self.price = price

    def __repr__(self):
        return f"Trade({self.timestamp}, {self.side}, qty={self.quantity}, price={self.price}, pos={self.position})"


def parse_trade_log(log_content: str) -> List[Trade]:
    """解析交易日志"""
    trades = []
    for line in log_content.strip().split("\n"):
        if "TRADE|" not in line:
            continue
        # 提取时间戳（前19个字符）
        timestamp = line[:19]
        # 分割TRADE部分
        trade_part = line.split("TRADE|")[1]
        fields = {}
        # 解析各个字段
        for field in trade_part.split("|"):
            key, value = field.split("=")
            # 去掉引号
            value = value.strip("'")
            fields[key] = value
        position = float(fields["position"])
        side = fields["side"]
        quantity = float(fields["quantity"])
        price = float(fields["price"])
        trades.append(Trade(timestamp, position, side, quantity, price))
    return trades


def calculate_pnl(trades: List[Trade]) -> Dict:
    """
    计算PNL - 使用mark-to-market方式

    注意：trade.position是成交前的仓位

    逻辑：
    - 每次交易时，前一个持仓按价格变化产生盈亏：prev_pos * (price - prev_price)
    - BUY操作：新仓位 = 旧仓位 + quantity
    - SELL操作：新仓位 = 旧仓位 - quantity
    """
    realized_pnl = Decimal("0")
    total_buy_cost = Decimal("0")
    total_sell_revenue = Decimal("0")
    total_buy_qty = Decimal("0")
    total_sell_qty = Decimal("0")

    trade_details = []
    prev_price = None

    for i, trade in enumerate(trades):
        # trade.position是成交前的仓位
        prev_position = Decimal(str(trade.position))
        quantity = Decimal(str(trade.quantity))
        price = Decimal(str(trade.price))

        # 计算本次交易的盈亏：前一个持仓在价格变化中的盈亏
        trade_pnl = Decimal("0")
        if prev_price is not None:
            trade_pnl = prev_position * (price - prev_price)
            realized_pnl += trade_pnl

        # 计算成交后的仓位
        if trade.side == "BUY":
            new_position = prev_position + quantity
            total_buy_cost += quantity * price
            total_buy_qty += quantity
        else:  # SELL
            new_position = prev_position - quantity
            total_sell_revenue += quantity * price
            total_sell_qty += quantity

        trade_details.append(
            {
                "timestamp": trade.timestamp,
                "side": trade.side,
                "quantity": float(quantity),
                "price": float(price),
                "trade_pnl": float(trade_pnl),
                "prev_position": float(prev_position),
                "new_position": float(new_position),
                "realized_pnl": float(realized_pnl),
            }
        )

        prev_price = price

    # 最终持仓就是最后一条交易后的持仓
    final_position = trade_details[-1]["new_position"] if trade_details else 0

    # Mark-to-market方式下，realized_pnl已经包含了所有盈亏
    # 没有未实现盈亏的概念，因为每次价格变化都已经mark到市场价了
    unrealized_pnl = Decimal("0")
    total_pnl = realized_pnl

    return {
        "realized_pnl": float(realized_pnl),
        "unrealized_pnl": float(unrealized_pnl),
        "total_pnl": float(total_pnl),
        "final_position": float(final_position),
        "total_buy_cost": float(total_buy_cost),
        "total_sell_revenue": float(total_sell_revenue),
        "total_buy_qty": float(total_buy_qty),
        "total_sell_qty": float(total_sell_qty),
        "total_trades": len(trades),
        "trade_details": trade_details,
    }


def print_summary(result: Dict):
    """打印PNL摘要"""
    print("=" * 80)
    print("交易统计摘要")
    print("=" * 80)
    print(f"总交易次数: {result['total_trades']}")
    print(f"总买入数量: {result['total_buy_qty']:.2f}")
    print(f"总卖出数量: {result['total_sell_qty']:.2f}")
    print(f"总买入成本: ${result['total_buy_cost']:.4f}")
    print(f"总卖出收益: ${result['total_sell_revenue']:.4f}")
    print()
    print(f"最终持仓: {result['final_position']:.2f}")
    print()
    print(f"总盈亏 (PNL): ${result['total_pnl']:.4f}")
    print("=" * 80)


def print_trade_details(result: Dict):
    """打印交易明细"""
    print("\n交易明细:")
    print("-" * 120)
    print(
        f"{'时间':<20} {'方向':<6} {'数量':<10} {'价格':<12} {'本次盈亏':<12} {'成交前':<10} {'成交后':<10} {'累计盈亏':<12}"
    )
    print("-" * 120)

    for detail in result["trade_details"]:
        print(
            f"{detail['timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<20} "
            f"{detail['side']:<6} "
            f"{detail['quantity']:<10.2f} "
            f"${detail['price']:<11.5f} "
            f"${detail['trade_pnl']:<11.4f} "
            f"{detail['prev_position']:<10.2f} "
            f"{detail['new_position']:<10.2f} "
            f"${detail['realized_pnl']:<11.4f}"
        )
    print("-" * 120)


if __name__ == "__main__":

    # 检查是否有stdin输入（管道输入）
    if not sys.stdin.isatty():
        # 从stdin读取（支持管道操作）
        log_content = sys.stdin.read()

        trades = parse_trade_log(log_content)
        if len(trades) == 0:
            print("未找到交易记录")
            sys.exit(1)

        print(f"解析到 {len(trades)} 条交易记录\n")

        result = calculate_pnl(trades)
        print_summary(result)
        print_trade_details(result)
    elif len(sys.argv) > 1:
        # 如果提供了文件路径参数
        log_file = sys.argv[1]
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()

        trades = parse_trade_log(log_content)
        print(f"解析到 {len(trades)} 条交易记录\n")

        result = calculate_pnl(trades)
        print_summary(result)
        print_trade_details(result)
    else:
        # 使用示例数据
        raise SystemExit("请提供日志文件路径或通过管道输入日志内容")