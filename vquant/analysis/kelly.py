#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Kelly Trader - Position sizing strategy based on Kelly Criterion
Dynamically adjusts position size based on win rate and risk/reward ratio
"""


import json
import logging
import pickle
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor


logger = logging.getLogger(__name__)


class KellyTrader(BasePredictor):
    """
    Kelly Criterion-based trading strategy
    
    Strategy Logic:
    - Calculate optimal position size using Kelly formula: f* = (bp - q) / b
    - Track historical win rate and average profit/loss ratio
    - Use fractional Kelly (e.g., 1/2 Kelly) to reduce risk
    - Adapt position size based on performance
    
    Kelly Formula:
    - f* = optimal fraction of capital to bet
    - b = profit/loss ratio (average win / average loss)
    - p = win rate (probability of winning)
    - q = 1 - p (probability of losing)
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDC",
        name: str = "default",
        config_dir: str = "config"
    ):
        """
        Initialize Kelly trader
        
        Args:
            symbol: Trading pair symbol
            name: Strategy name (used for state file naming)
            config_dir: Directory containing config files
        """
        super().__init__(symbol, name)
        
        # Load configuration from file
        config = self._load_config(self.name, self.symbol, config_dir)

        self.base_volume = config.get('base_volume', 0.1)
        self.max_volume = config.get('max_volume', 1.0)
        self.min_volume = config.get('min_volume', 0.01)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Use 1/4 Kelly for safety
        self.profit_target = config.get('profit_target', 1.0)  # % profit to take
        self.stop_loss = config.get('stop_loss', -1.0)  # % stop loss
        self.min_trades_for_kelly = config.get('min_trades_for_kelly', 10)  # Min trades to calculate Kelly
        self.account_balance = config.get('account_balance', 10000.0)
        self.max_position_pct = config.get('max_position_pct', 0.5)  # Max 50% of balance
        self.min_trade_interval = config.get('min_trade_interval', 300)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.1)  # 10% change triggers rebalance
        
        self.state_file = Path(f"data/kelly_state_{name}.json")
        
        logger.info(
            f"Kelly trader initialized: base={self.base_volume}, "
            f"kelly_fraction={self.kelly_fraction}, min_trades={self.min_trades_for_kelly}"
        )
        logger.info(
            f"Targets: profit={self.profit_target}%, stop_loss={self.stop_loss}%"
        )
        
        # ML model for win probability prediction
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.model_path = Path("data") / f'kelly_model_{name}.pkl'
        self.scaler_path = Path("data") / f'kelly_scaler_{name}.pkl'
        
        # Training configuration
        self.min_samples_for_training = config.get('min_samples_for_training', 100)
        self.retrain_interval = config.get('retrain_interval', 50)
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Load trained model if available
        self._load_model()
    
    def _load_config(self, name: str, symbol: str, config_dir: str) -> Dict[str, Any]:
        """
        Load Kelly trader configuration from file
        
        Args:
            name: Strategy name
            symbol: Trading pair symbol
            config_dir: Directory containing config files
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: Config file does not exist
        """
        config_path = Path(config_dir) / f"kelly_{name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Kelly trader config file not found: {config_path}\n"
                f"Please create config file for {symbol}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            assert symbol == config["symbol"]
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file JSON format error: {e}")
    
    def _extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from market data for ML model.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Feature array
        """
        features = []
        
        # Price features
        current_price = market_data.get('current_price', 0)
        features.append(current_price)
        
        # Price changes
        if 'klines' in market_data and len(market_data['klines']) > 0:
            closes = [k[4] for k in market_data['klines']]  # Close prices
            if len(closes) >= 5:
                # Short-term momentum (5 periods)
                features.append((closes[-1] - closes[-5]) / closes[-5] * 100)
            else:
                features.append(0.0)
                
            if len(closes) >= 20:
                # Medium-term momentum (20 periods)
                features.append((closes[-1] - closes[-20]) / closes[-20] * 100)
                # Volatility (std of returns)
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                features.append(np.std(returns) * 100)
            else:
                features.append(0.0)
                features.append(0.0)
                
            # Volume features
            if len(market_data['klines']) >= 5:
                volumes = [k[5] for k in market_data['klines'][-5:]]  # Last 5 volumes
                current_volume = volumes[-1]
                avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else current_volume
                features.append(current_volume / avg_volume if avg_volume > 0 else 1.0)
            else:
                features.append(1.0)
        else:
            features.extend([0.0, 0.0, 0.0, 1.0])  # Default values
            
        # Funding rate (if available)
        funding_rate = market_data.get('funding_rate', 0)
        features.append(funding_rate * 10000)  # Scale up
        
        # Technical indicators
        if 'klines' in market_data and len(market_data['klines']) >= 20:
            closes = [k[4] for k in market_data['klines']]
            
            # RSI-like indicator
            gains = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
            losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100 if avg_gain > 0 else 50
            features.append(rsi)
            
            # MA deviation
            ma20 = np.mean(closes[-20:])
            features.append((closes[-1] - ma20) / ma20 * 100)
        else:
            features.extend([50.0, 0.0])  # Neutral RSI and MA deviation
            
        return np.array(features).reshape(1, -1)
    
    def _load_state(self) -> Dict[str, Any]:
        """
        Load trading state from file
        
        Returns:
            State dictionary with trade history and statistics
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logger.info(f"Loaded trading state from {self.state_file}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}, using default state")
        
        # Default state
        return {
            "symbol": self.symbol,
            "current_position": 0.0,
            "entry_price": None,
            "last_trade_time": None,
            "last_trade_action": None,
            "last_trade_volume": None,
            
            # Kelly calculation statistics
            "total_trades": 0,
            "winning_trades": 0,
            "total_profit": 0.0,  # Sum of all profits
            "total_loss": 0.0,    # Sum of all losses (absolute values)
            "win_rate": 0.0,
            "profit_loss_ratio": 1.0,
            "kelly_percentage": 0.0,
            
            # Trade history (last 100 trades for rolling statistics)
            "trade_history": [],
            
            # Training data buffer for ML model
            "training_buffer": [],
            "last_training_count": 0,
            
            # Performance tracking
            "total_pnl": 0.0,
            "peak_pnl": 0.0,
            "max_drawdown": 0.0,
        }
    
    def _save_state(self):
        """Save trading state to file"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved trading state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")
    
    def _load_model(self) -> None:
        """Load trained ML model from disk."""
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded ML model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            self.model = None
            self.scaler = None
    
    def _save_model(self) -> None:
        """Save trained ML model to disk."""
        try:
            if self.model is not None and self.scaler is not None:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info(f"Saved ML model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _train_model(self) -> None:
        """Train ML model to predict win probability."""
        training_buffer = self.state.get('training_buffer', [])
        
        if len(training_buffer) < self.min_samples_for_training:
            logger.info(
                f"Not enough samples for training: "
                f"{len(training_buffer)}/{self.min_samples_for_training}"
            )
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            for sample in training_buffer:
                X.append(sample['features'])
                y.append(1 if sample['profitable'] else 0)
            
            X = np.vstack(X)
            y = np.array(y)
            
            # Check class balance
            win_rate = np.mean(y)
            logger.info(
                f"Training with {len(y)} samples, "
                f"historical win rate: {win_rate:.2%}"
            )
            
            # Initialize and fit scaler
            if self.scaler is None:
                self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize and train model
            if self.model is None:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight='balanced'  # Handle imbalanced data
                )
            
            self.model.fit(X_scaled, y)
            
            # Calculate training accuracy
            train_score = self.model.score(X_scaled, y)
            logger.info(f"Model trained, accuracy: {train_score:.2%}")
            
            # Save model
            self._save_model()
            
            # Update last training count
            self.state['last_training_count'] = self.state['total_trades']
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
    
    def _predict_win_probability(self, market_data: Dict[str, Any]) -> float:
        """Predict win probability using ML model.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Predicted win probability (0-1)
        """
        # Extract features
        features = self._extract_features(market_data)
        
        # If model not trained, use historical win rate or default
        if self.model is None or self.scaler is None:
            # Use historical win rate if available
            if self.state['total_trades'] >= 10:
                historical_win_rate = self.state['win_rate']
                logger.info(f"Model not ready, using historical win rate: {historical_win_rate:.2%}")
                return historical_win_rate
            else:
                # Conservative default
                default_prob = 0.52
                logger.info(f"No model or history, using default: {default_prob:.2%}")
                return default_prob
        
        try:
            # Scale features and predict
            features_scaled = self.scaler.transform(features)
            proba = self.model.predict_proba(features_scaled)[0]
            win_prob = proba[1]  # Probability of class 1 (win)
            
            # Clip to reasonable range
            win_prob = np.clip(win_prob, 0.35, 0.65)
            
            logger.info(f"ML predicted win probability: {win_prob:.2%}")
            return win_prob
            
        except Exception as e:
            logger.error(f"Prediction error: {e}, using historical rate")
            return self.state.get('win_rate', 0.52)
            logger.error(f"Failed to save state file: {e}")
    
    def _can_trade(self) -> bool:
        """
        Check if enough time has passed since last trade
        
        Returns:
            True if can trade, False otherwise
        """
        if self.state["last_trade_time"] is None:
            return True
        
        last_time = datetime.fromisoformat(self.state["last_trade_time"])
        current_time = datetime.now()
        elapsed_seconds = (current_time - last_time).total_seconds()
        
        if elapsed_seconds < self.min_trade_interval:
            logger.info(
                f"Too soon to trade. Last trade: {elapsed_seconds:.0f}s ago, "
                f"minimum interval: {self.min_trade_interval}s"
            )
            return False
        
        return True
    
    def _calculate_kelly_percentage(self, market_data: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Kelly formula: f* = (bp - q) / b
        where:
        - b = profit/loss ratio (average win / average loss)
        - p = win rate (predicted by ML model or historical)
        - q = 1 - p
        
        Args:
            market_data: Market data for ML prediction (optional)
        
        Returns:
            Kelly percentage (fraction of capital to risk)
        """
        # Get win rate from ML prediction or historical data
        if market_data is not None:
            # Use ML model to predict win probability
            win_rate = self._predict_win_probability(market_data)
        else:
            # Fall back to historical win rate
            if self.state["total_trades"] < self.min_trades_for_kelly:
                logger.info(
                    f"Only {self.state['total_trades']} trades, need {self.min_trades_for_kelly} "
                    f"for Kelly calculation. Using conservative estimate."
                )
                win_rate = 0.52  # Slightly positive edge
            else:
                win_rate = self.state["win_rate"]
        
        # Calculate profit/loss ratio from historical data
        pl_ratio = self.state.get("profit_loss_ratio", 1.0)
        
        # If no history yet, estimate from configured targets
        if self.state["total_trades"] == 0:
            pl_ratio = abs(self.profit_target / self.stop_loss) if self.stop_loss != 0 else 1.0
        
        lose_rate = 1 - win_rate
        
        # Kelly formula
        if pl_ratio <= 0 or win_rate <= 0.5:
            logger.warning(f"Unfavorable Kelly parameters (win_rate={win_rate:.2%}, pl_ratio={pl_ratio:.2f}), using minimum")
            return 0.0
        
        kelly = (pl_ratio * win_rate - lose_rate) / pl_ratio
        
        # Apply Kelly fraction for safety (typically 1/4 or 1/2 Kelly)
        fractional_kelly = kelly * self.kelly_fraction
        
        # Constrain to reasonable bounds
        kelly_pct = max(0.0, min(fractional_kelly, self.max_position_pct))
        
        logger.info(
            f"Kelly calculation: win_rate={win_rate:.2%} (ML predicted), pl_ratio={pl_ratio:.2f}, "
            f"full_kelly={kelly:.2%}, fractional({self.kelly_fraction})={kelly_pct:.2%}"
        )
        
        return kelly_pct
    
    def _calculate_position_size(
        self, 
        current_price: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate position size based on Kelly percentage
        
        Args:
            current_price: Current market price
            market_data: Market data for ML prediction (optional)
            
        Returns:
            Position size in base currency
        """
        kelly_pct = self._calculate_kelly_percentage(market_data)
        
        if kelly_pct <= 0:
            # Use base volume if Kelly not applicable
            return self.base_volume
        
        # Calculate position size based on account balance and Kelly percentage
        position_value = self.account_balance * kelly_pct
        position_size = position_value / current_price
        
        # Apply volume constraints
        position_size = max(self.min_volume, min(position_size, self.max_volume))
        
        logger.info(
            f"Position sizing: kelly={kelly_pct:.2%}, balance=${self.account_balance:.2f}, "
            f"position_value=${position_value:.2f}, size={position_size:.4f}"
        )
        
        return position_size
    
    def _update_statistics(self, pnl_pct: float):
        """
        Update win rate and profit/loss ratio statistics
        
        Args:
            pnl_pct: Realized P&L percentage
        """
        self.state["total_trades"] += 1
        
        # Record trade in history (keep last 100)
        self.state["trade_history"].append({
            "pnl_pct": pnl_pct,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.state["trade_history"]) > 100:
            self.state["trade_history"].pop(0)
        
        # Update win/loss counts
        is_profitable = pnl_pct > 0
        if is_profitable:
            self.state["winning_trades"] += 1
            self.state["total_profit"] += pnl_pct
        else:
            self.state["total_loss"] += abs(pnl_pct)
        
        # Add training sample if we have entry features
        entry_features = self.state.get("entry_features")
        if entry_features is not None:
            training_buffer = self.state.get("training_buffer", [])
            training_buffer.append({
                'features': entry_features,
                'profitable': is_profitable,
                'pnl': pnl_pct,
                'timestamp': datetime.now().isoformat()
            })
            # Keep only last 200 samples
            if len(training_buffer) > 200:
                training_buffer.pop(0)
            self.state["training_buffer"] = training_buffer
            
            logger.info(f"Added training sample, buffer size: {len(training_buffer)}")
        
        # Calculate win rate
        self.state["win_rate"] = self.state["winning_trades"] / self.state["total_trades"]
        
        # Calculate profit/loss ratio
        avg_win = (self.state["total_profit"] / self.state["winning_trades"] 
                   if self.state["winning_trades"] > 0 else 0)
        losing_trades = self.state["total_trades"] - self.state["winning_trades"]
        avg_loss = (self.state["total_loss"] / losing_trades 
                    if losing_trades > 0 else 1)
        self.state["profit_loss_ratio"] = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Update total P&L
        self.state["total_pnl"] += pnl_pct
        
        # Update peak and drawdown
        if self.state["total_pnl"] > self.state["peak_pnl"]:
            self.state["peak_pnl"] = self.state["total_pnl"]
        drawdown = self.state["total_pnl"] - self.state["peak_pnl"]
        if drawdown < self.state["max_drawdown"]:
            self.state["max_drawdown"] = drawdown
        
        # Recalculate Kelly percentage
        self.state["kelly_percentage"] = self._calculate_kelly_percentage()
        
        logger.info(
            f"Stats updated: trades={self.state['total_trades']}, "
            f"win_rate={self.state['win_rate']:.2%}, "
            f"pl_ratio={self.state['profit_loss_ratio']:.2f}, "
            f"kelly={self.state['kelly_percentage']:.2%}"
        )
        
        # Check if we should retrain the model
        training_buffer = self.state.get("training_buffer", [])
        trades_since_training = self.state['total_trades'] - self.state.get('last_training_count', 0)
        
        if (len(training_buffer) >= self.min_samples_for_training and 
            trades_since_training >= self.retrain_interval):
            logger.info(
                f"Triggering model retraining "
                f"({trades_since_training} trades since last training)"
            )
            self._train_model()
    
    def _calculate_pnl(self, current_price: float) -> Optional[float]:
        """
        Calculate unrealized P&L percentage
        
        Args:
            current_price: Current market price
            
        Returns:
            P&L percentage, or None if no position
        """
        if self.state["entry_price"] is None or self.state["current_position"] == 0:
            return None
        
        entry_price = self.state["entry_price"]
        position = self.state["current_position"]
        
        if position > 0:  # Long position
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # Short position
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        return pnl_pct
    
    def _should_rebalance(self, current_price: float) -> bool:
        """
        Check if position should be rebalanced based on Kelly percentage change
        
        Args:
            current_price: Current market price
            
        Returns:
            True if should rebalance
        """
        if self.state["current_position"] == 0:
            return False
        
        # Calculate current position value as % of balance
        position_value = abs(self.state["current_position"]) * current_price
        current_pct = position_value / self.account_balance
        
        # Calculate optimal Kelly percentage
        optimal_pct = self._calculate_kelly_percentage()
        
        if optimal_pct <= 0:
            return False
        
        # Check if deviation exceeds threshold
        deviation = abs(current_pct - optimal_pct) / optimal_pct
        
        if deviation > self.rebalance_threshold:
            logger.info(
                f"Rebalance needed: current={current_pct:.2%}, "
                f"optimal={optimal_pct:.2%}, deviation={deviation:.2%}"
            )
            return True
        
        return False
    
    def _record_trade(self, action: str, price: float, volume: float, reasoning: str):
        """
        Record trade in state
        
        Args:
            action: Trade action (buy/sell/close)
            price: Trade price
            volume: Trade volume
            reasoning: Trade reasoning
        """
        old_position = self.state["current_position"]
        
        if action == "buy":
            new_position = old_position + volume
            if self.state["entry_price"] is None:
                self.state["entry_price"] = price
            else:
                # Weighted average entry price
                total_cost = old_position * self.state["entry_price"] + volume * price
                self.state["entry_price"] = total_cost / new_position if new_position != 0 else price
            self.state["current_position"] = new_position
            
        elif action == "sell":
            new_position = old_position - volume
            if old_position > 0 and new_position <= 0:
                # Closing long or flipping to short
                if self.state["entry_price"]:
                    closed_volume = min(volume, old_position)
                    pnl = ((price - self.state["entry_price"]) / self.state["entry_price"]) * 100
                    self._update_statistics(pnl)
                
                if new_position < 0:
                    self.state["entry_price"] = price
                else:
                    self.state["entry_price"] = None
            elif old_position < 0:
                # Adding to short
                if self.state["entry_price"] is None:
                    self.state["entry_price"] = price
                else:
                    total_cost = abs(old_position) * self.state["entry_price"] + volume * price
                    self.state["entry_price"] = total_cost / abs(new_position) if new_position != 0 else price
            
            self.state["current_position"] = new_position
            
        elif action == "close":
            if old_position != 0 and self.state["entry_price"]:
                if old_position > 0:
                    pnl = ((price - self.state["entry_price"]) / self.state["entry_price"]) * 100
                else:
                    pnl = ((self.state["entry_price"] - price) / self.state["entry_price"]) * 100
                self._update_statistics(pnl)
            
            self.state["current_position"] = 0.0
            self.state["entry_price"] = None
        
        self.state["last_trade_time"] = datetime.now().isoformat()
        self.state["last_trade_action"] = action
        self.state["last_trade_volume"] = volume
        
        self._save_state()
        logger.info(
            f"Recorded trade: {action} {volume:.4f} @ ${price:.2f} | "
            f"Position: {self.state['current_position']:.4f}"
        )
    
    def prepare_data(self, df, df_display, ma_dict, ma_dict_display, stats, args) -> Tuple[Optional[str], Optional[bytes]]:
        """
        Kelly trader doesn't need chart generation
        
        Returns:
            (None, None) as no chart is needed
        """
        logger.info("Kelly trader mode: skipping chart generation")
        return None, None
    
    def analyze(self, stats: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analyze using Kelly Criterion strategy
        
        Args:
            stats: Statistics dictionary
            **kwargs: Additional parameters
            
        Returns:
            Trading signal
        """
        current_price = stats.get('current_price')
        
        return self.generate_signal(
            current_price=current_price,
            stats=stats
        )
    
    def generate_output(self, result: Dict[str, Any], stats: Dict[str, Any], args) -> Dict[str, Any]:
        """
        Generate standardized output for Kelly trader
        
        Args:
            result: Signal result from analyze()
            stats: Statistics dictionary
            args: Command line arguments
            
        Returns:
            Standardized output dictionary
        """
        action_to_position = {
            'buy': 1.0,
            'sell': -1.0,
            'close': 0.0,
            'rebalance': 0.5,
            'hold': 0.0
        }
        
        return {
            'symbol': result['symbol'],
            'position': action_to_position.get(result['action'], 0.0),
            'confidence': 'high' if result['action'] in ['buy', 'sell', 'close'] else 'medium',
            'current_price': result['current_price'],
            'action': result['action'],
            'volume': result['volume'],
            'reasoning': result['reasoning'],
            'analysis_type': 'kelly',
            'current_position': result.get('current_position'),
            'kelly_percentage': result.get('kelly_percentage'),
            'win_rate': result.get('win_rate'),
            'pl_ratio': result.get('pl_ratio'),
        }
    
    def generate_signal(
        self,
        current_price: float,
        stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signal based on Kelly Criterion
        
        Args:
            current_price: Current market price
            stats: Optional market statistics (used for ML prediction)
            
        Returns:
            Trade signal dictionary
        """
        logger.info(
            f"Generating signal for {self.symbol} @ ${current_price:.2f} | "
            f"Position: {self.state['current_position']:.4f}, "
            f"Kelly: {self.state['kelly_percentage']:.2%}"
        )
        
        # Extract features for ML prediction (if stats provided)
        current_features = None
        if stats is not None:
            current_features = self._extract_features(stats).flatten()
        
        # Check if we can trade
        if not self._can_trade():
            return {
                "action": "hold",
                "volume": 0.0,
                "symbol": self.symbol,
                "current_price": current_price,
                "current_position": self.state["current_position"],
                "reasoning": "Minimum trade interval not reached",
                "kelly_percentage": self.state["kelly_percentage"],
                "win_rate": self.state["win_rate"],
                "pl_ratio": self.state["profit_loss_ratio"],
            }
        
        # Calculate current P&L
        pnl_pct = self._calculate_pnl(current_price)
        
        # Calculate optimal position size (with ML prediction if available)
        optimal_size = self._calculate_position_size(current_price, stats)
        
        signal = {
            "action": "hold",
            "volume": 0.0,
            "symbol": self.symbol,
            "current_price": current_price,
            "current_position": self.state["current_position"],
            "kelly_percentage": self.state["kelly_percentage"],
            "win_rate": self.state["win_rate"],
            "pl_ratio": self.state["profit_loss_ratio"],
            "reasoning": ""
        }
        
        # Check if we have an open position
        if self.state["current_position"] != 0 and pnl_pct is not None:
            # Check for profit target
            if pnl_pct >= self.profit_target:
                signal["action"] = "close"
                signal["volume"] = abs(self.state["current_position"])
                signal["reasoning"] = (
                    f"Taking profit: P&L {pnl_pct:.2f}% >= target {self.profit_target}% "
                    f"(entry: ${self.state['entry_price']:.2f})"
                )
                self._record_trade("close", current_price, signal["volume"], signal["reasoning"])
                return signal
            
            # Check for stop loss
            if pnl_pct <= self.stop_loss:
                signal["action"] = "close"
                signal["volume"] = abs(self.state["current_position"])
                signal["reasoning"] = (
                    f"Stop loss: P&L {pnl_pct:.2f}% <= {self.stop_loss}% "
                    f"(entry: ${self.state['entry_price']:.2f})"
                )
                self._record_trade("close", current_price, signal["volume"], signal["reasoning"])
                return signal
            
            # Check if rebalancing is needed
            if self._should_rebalance(current_price):
                position_value = abs(self.state["current_position"]) * current_price
                current_pct = position_value / self.account_balance
                target_pct = self.state["kelly_percentage"]
                
                if current_pct > target_pct:
                    # Reduce position
                    reduce_size = abs(self.state["current_position"]) * (1 - target_pct / current_pct)
                    signal["action"] = "sell" if self.state["current_position"] > 0 else "buy"
                    signal["volume"] = reduce_size
                    signal["reasoning"] = (
                        f"Rebalance down: current={current_pct:.2%} > optimal={target_pct:.2%}, "
                        f"reducing by {reduce_size:.4f}"
                    )
                else:
                    # Increase position
                    increase_size = optimal_size - abs(self.state["current_position"])
                    signal["action"] = "buy" if self.state["current_position"] > 0 else "sell"
                    signal["volume"] = increase_size
                    signal["reasoning"] = (
                        f"Rebalance up: current={current_pct:.2%} < optimal={target_pct:.2%}, "
                        f"adding {increase_size:.4f}"
                    )
                
                self._record_trade(signal["action"], current_price, signal["volume"], signal["reasoning"])
                return signal
            
            # Hold position
            signal["reasoning"] = (
                f"Holding {self.state['current_position']:.4f}, "
                f"P&L: {pnl_pct:.2f}% (Kelly: {self.state['kelly_percentage']:.2%})"
            )
        else:
            # No position - open new one based on Kelly
            if optimal_size >= self.min_volume:
                # Save entry features for training
                if current_features is not None:
                    self.state["entry_features"] = current_features.tolist()
                
                signal["action"] = "buy"
                signal["volume"] = optimal_size
                signal["reasoning"] = (
                    f"Opening position: Kelly={self.state['kelly_percentage']:.2%}, "
                    f"size={optimal_size:.4f} (win_rate={self.state['win_rate']:.2%}, "
                    f"pl_ratio={self.state['profit_loss_ratio']:.2f})"
                )
                self._record_trade("buy", current_price, optimal_size, signal["reasoning"])
            else:
                signal["reasoning"] = (
                    f"Kelly size {optimal_size:.4f} below minimum {self.min_volume}, "
                    f"waiting for better statistics"
                )
        
        logger.info(f"Signal: {signal['action'].upper()} - {signal['reasoning']}")
        
        return signal
