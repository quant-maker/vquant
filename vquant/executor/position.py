#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os
import json
import logging


logger = logging.getLogger(__name__)


class PositionManager:
    """Position Manager - Manage strategy positions and order states"""
    
    def __init__(self, cli, strategy_id: str = 'default'):
        """
        Initialize position manager
        Args:
            cli: Exchange client (e.g., USDM client)
            strategy_id: Strategy identifier
        """
        self.cli = cli
        self.strategy_id = strategy_id
        
        # Order state file for tracking pending orders
        self.state_dir = 'logs/orders'
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_file = os.path.join(self.state_dir, f'{strategy_id}_order_state.json')
        
        logger.debug(f"PositionManager initialized for strategy '{strategy_id}'")
    
    def get_current_position(self, symbol: str) -> float:
        """
        Get current filled position for this strategy
        Will check and handle previous order if exists:
        - If order filled: update position and clear order info
        - If order not filled: cancel order and restore position
        - If partially filled: adjust position accordingly
        
        Args:
            symbol: Trading symbol
        Returns:
            Position amount (positive for long, negative for short)
        """
        try:
            state = self._load_order_state()
            if not state or state.get('symbol') != symbol:
                logger.debug(f"No position record for {symbol}, assuming 0")
                return 0
            
            # If there's a pending order, check its status
            if 'order_id' in state:
                order_id = state['order_id']
                logger.info(f"Checking previous order: {order_id}")
                try:
                    # Query order status
                    order_info = self.cli.get_order(symbol=symbol, orderId=order_id)
                    status = order_info['status']
                    logger.info(f"Previous order status: {status}")
                    
                    if status == 'FILLED':
                        # Order filled, update position by adding/subtracting order quantity
                        logger.info("Previous order filled ✓")
                        old_pos = float(state.get('current_position', 0))
                        quantity = float(state['quantity'])
                        side = state['side']
                        
                        # Calculate new position after filled
                        if side == 'BUY':
                            new_pos = old_pos + quantity
                        else:  # SELL
                            new_pos = old_pos - quantity
                        logger.info(f"Position updated: {old_pos} -> {new_pos} (filled {side} {quantity})")
                        self._save_position_only(symbol, new_pos)
                        return new_pos
                    elif status in ['NEW', 'PARTIALLY_FILLED']:
                        # Order not filled or partially filled, cancel it
                        logger.info(f"Canceling previous order: {order_id}")
                        self.cli.cancel_order(symbol=symbol, orderId=order_id)
                        # Calculate actual position based on filled quantity
                        old_pos = float(state.get('current_position', 0))
                        orig_qty = float(state['quantity'])
                        filled_qty = float(order_info.get('executedQty', 0))
                        side = state['side']
                        if filled_qty > 0:
                            # Partially filled, adjust position by filled quantity
                            if side == 'BUY':
                                actual_pos = old_pos + filled_qty
                            else:  # SELL
                                actual_pos = old_pos - filled_qty
                            logger.info(f"Partially filled: {filled_qty}/{orig_qty}, position: {old_pos} -> {actual_pos}")
                        else:
                            # Not filled at all, position unchanged
                            actual_pos = old_pos
                            logger.info(f"Order not filled, position remains: {actual_pos}")
                        self._save_position_only(symbol, actual_pos)
                        return actual_pos
                    else:
                        # Order in other status (CANCELED, REJECTED, EXPIRED)
                        logger.warning(f"Previous order in status {status}, attempting to cancel")
                        try:
                            self.cli.cancel_order(symbol=symbol, orderId=order_id)
                            logger.info(f"Successfully canceled order {order_id}")
                        except Exception as cancel_err:
                            logger.warning(f"Failed to cancel order (may already be canceled): {cancel_err}")
                        
                        # Keep position, only clear order info
                        old_pos = float(state.get('current_position', 0))
                        logger.info(f"Clearing order info, keeping position: {old_pos}")
                        self._save_position_only(symbol, old_pos)
                        return old_pos
                except Exception as e:
                    logger.exception(f"Failed to check previous order: {e}")
                    # Fall through to return current position from state
            # No pending order, return current position
            position = float(state.get('current_position', 0))
            logger.debug(f"Current strategy position for {symbol}: {position}")
            return position
        except Exception as e:
            logger.exception(f"Failed to get position from state: {e}")
            return 0
    
    def _save_order_state(self, symbol: str, order_id: int, side: str, quantity: float, current_position: float):
        """
        Save order state to file
        Args:
            symbol: Trading symbol
            order_id: Order ID
            side: Order side (BUY/SELL)
            quantity: Order quantity
            current_position: Current position BEFORE this order (not after)
        """
        state = {
            'symbol': symbol,
            'order_id': order_id,
            'side': side,
            'quantity': quantity,
            'current_position': current_position
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
            logger.debug(f"Saved order state: {state}")
        except Exception as e:
            logger.exception(f"Failed to save order state: {e}")
    
    def _load_order_state(self) -> dict:
        """
        Load order state from file
        Returns:
            Order state dict or None if not exists
        """
        if not os.path.exists(self.state_file):
            return None
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.debug(f"Loaded order state: {state}")
            return state
        except Exception as e:
            logger.exception(f"Failed to load order state: {e}")
            return None
    
    def _save_position_only(self, symbol: str, position: float):
        """
        Save only position state (no pending order)
        Args:
            symbol: Trading symbol
            position: Current position
        """
        state = {
            'symbol': symbol,
            'current_position': position
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved position state: {state}")
        except Exception as e:
            logger.exception(f"Failed to save position state: {e}")
    
    def handle_previous_order(self, symbol: str) -> tuple:
        """
        Deprecated: This method is now integrated into get_current_position()
        Kept for backward compatibility, just returns success
        
        Returns:
            tuple: (should_continue: bool, adjustment: float)
        """
        logger.debug("handle_previous_order called but order handling is now in get_current_position")
        return True, 0
    
    def save_new_order(self, symbol: str, order_id: int, side: str, quantity: float, current_position: float):
        """
        Save newly placed order state
        Args:
            symbol: Trading symbol
            order_id: Order ID
            side: Order side (BUY/SELL)
            quantity: Order quantity
            current_position: Current position BEFORE placing this order
        """
        self._save_order_state(symbol, order_id, side, quantity, current_position)
        logger.info(f"Saved order state: {side} {quantity} {symbol} (ID: {order_id}), current position: {current_position}")
    
    def get_order_state(self) -> dict:
        """
        Get current order state
        Returns:
            Order state dict or None
        """
        return self._load_order_state()
    
    def set_initial_position(self, symbol: str, position: float):
        """
        Set initial position for this strategy (use when starting a strategy)
        Args:
            symbol: Trading symbol
            position: Initial position amount
        """
        logger.info(f"Setting initial position for {symbol}: {position}")
        self._save_position_only(symbol, position)
