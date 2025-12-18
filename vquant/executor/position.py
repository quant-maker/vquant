#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os
import json
import logging


logger = logging.getLogger(__name__)


class PositionManager:
    """Position Manager - Manage strategy positions and order states"""
    
    def __init__(self, cli, name: str = 'default'):
        """
        Initialize position manager
        Args:
            cli: Exchange client (e.g., USDM client)
            name: Strategy name (unique identifier for this strategy)
        """
        self.cli = cli
        self.name = name
        # Order state file for tracking pending orders
        self.state_dir = 'logs/orders'
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_file = os.path.join(self.state_dir, f'{name}_order_state.json')
        # Check for duplicate strategy instances
        if os.path.exists(self.state_file):
            logger.error("Please use a different strategy name (--name) or stop the existing instance.")
            raise RuntimeError(f"Strategy '{name}' is already running. Use a different name.")
        logger.debug(f"PositionManager initialized for strategy '{name}'")
    
    def get_current_position(self) -> float:
        """
        Get current filled position for this strategy
        Will check and handle previous order if exists:
        - If order filled: update position and mark as processed
        - If order not filled: cancel order and restore position
        - If partially filled: adjust position accordingly
        Returns:
            float: Position amount (positive for long, negative for short)
                   Returns 0 if no state exists
        """
        state = self._load_order_state()
        if not state:
            logger.debug("No position record found, assuming 0")
            return 0
        # If there's a pending order (not yet processed), check its status
        if 'order_id' in state and not state.get('processed', False):
            symbol = state['symbol']
            order_id = state['order_id']
            logger.info(f"Checking previous order: {order_id}")
            try:
                # Query order status
                order_info = self.cli.get_order(symbol=symbol, orderId=order_id)
                status = order_info['status']
                if status == 'FILLED':
                    # Order filled, update position by adding/subtracting order quantity
                    logger.info("Previous order filled ✓")
                    old_pos = float(state.get('position', 0))
                    quantity = float(state['quantity'])
                    side = state['side']
                    # Calculate new position after filled
                    if side == 'BUY':
                        new_pos = old_pos + quantity
                    else:  # SELL
                        new_pos = old_pos - quantity
                    logger.info(f"Position updated: {old_pos} -> {new_pos} (filled {side} {quantity})")
                    self._mark_order_processed(state, new_pos)
                    return new_pos
                elif status in ['NEW', 'PARTIALLY_FILLED']:
                    # Order not filled or partially filled, cancel it
                    logger.info(f"Canceling previous order: {order_id}")
                    self.cli.cancel_order(symbol=symbol, orderId=order_id)
                    # Calculate actual position based on filled quantity
                    old_pos = float(state.get('position', 0))
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
                    self._mark_order_processed(state, actual_pos)
                    return actual_pos
                else:
                    # Order in other status (CANCELED, REJECTED, EXPIRED)
                    logger.warning(f"Previous order in status {status}, attempting to cancel")
                    try:
                        self.cli.cancel_order(symbol=symbol, orderId=order_id)
                        logger.info(f"Successfully canceled order {order_id}")
                    except Exception as cancel_err:
                        logger.warning(f"Failed to cancel order (may already be canceled): {cancel_err}")

                    # Keep position and order info, mark as processed
                    old_pos = float(state.get('position', 0))
                    logger.info(f"Marking order as processed, keeping position: {old_pos}")
                    self._mark_order_processed(state, old_pos)
                    return old_pos
            except Exception as e:
                logger.exception(f"Failed to check previous order: {e}")
                # Fall through to return current position from state
        # No pending order, return current position
        position = float(state.get('position', 0))
        logger.debug(f"Current strategy position for {symbol}: {position}")
        return position
    
    def _save_order_state(self, symbol: str, order_id: int, side: str, quantity: float, position: float):
        """
        Save order state to file
        Args:
            symbol: Trading symbol
            order_id: Order ID
            side: Order side (BUY/SELL)
            quantity: Order quantity
            position: Current position BEFORE this order (not after)
        """
        state = {
            'symbol': symbol,
            'order_id': order_id,
            'side': side,
            'quantity': quantity,
            'position': position,
            'processed': False  # Mark as unprocessed when placing new order
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
            self._mark_order_processed({}, 0)  # Ensure file exists
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.debug(f"Loaded order state: {state}")
            return state
        except Exception as e:
            logger.exception(f"Failed to load order state: {e}")
            return None
    
    def _mark_order_processed(self, state: dict, position: float):
        """
        Mark order as processed and update position
        Keep order info for debugging, just mark it as processed
        Args:
            state: Current state dict
            position: Updated position
        """
        state['position'] = position
        state['processed'] = True
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
            logger.debug(f"Marked order as processed: {state}")
        except Exception as e:
            logger.exception(f"Failed to mark order as processed: {e}")
    
    def save_new_order(self, symbol: str, order_id: int, side: str, quantity: float, position: float):
        """
        Save newly placed order state
        Args:
            symbol: Trading symbol
            order_id: Order ID
            side: Order side (BUY/SELL)
            quantity: Order quantity
            position: Current position BEFORE placing this order
        """
        self._save_order_state(symbol, order_id, side, quantity, position)
        logger.info(f"Saved order state: {side} {quantity} {symbol} (ID: {order_id}), position: {position}")
    
    def get_order_state(self) -> dict:
        """
        Get current order state
        Returns:
            Order state dict or None
        """
        return self._load_order_state()