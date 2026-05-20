from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
from loguru import logger
from strategy_engine.base_strategy import BaseStrategy, TradeSignal, Signal


"""
TechMomentumEmaCross Strategy

A momentum trend-following strategy for large-cap US tech stocks that enters long 
positions when price crosses above both fast and slow EMAs with confirming RSI and 
MACD signals. The strategy captures medium-term uptrends while filtering for bullish 
market regimes.

Entry Criteria:
- Close price crosses above EMA-9
- Close price is above EMA-21
- RSI-14 is between 50 and 70 (momentum zone)
- MACD histogram turns positive
- Daily close is above SMA-200 (regime filter)
- ADX-14 is above 20 (trending market)

Exit Criteria:
- Stop loss: Price falls below entry - 1.5× ATR-14
- Take profit: Price reaches entry + 3.0× ATR-14
- RSI-14 exceeds 75 (overbought)
- Close price crosses below EMA-21
"""


class TechMomentumEmaCross(BaseStrategy):
    """
    Tech Momentum EMA Cross Strategy
    
    Implements a dual EMA crossover with RSI and MACD confirmation for
    large-cap tech stocks. Uses ATR-based stops and targets with regime filtering.
    """
    
    def __init__(self, parameters: dict = None):
        """
        Initialize the strategy with parameters.
        
        Args:
            parameters: Dictionary containing strategy parameters
        """
        default_parameters = {
            "ema_fast_period": 9,
            "ema_slow_period": 21,
            "rsi_period": 14,
            "rsi_entry_min": 50,
            "rsi_entry_max": 70,
            "rsi_exit_threshold": 75,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "atr_period": 14,
            "atr_stop_multiple": 1.5,
            "atr_target_multiple": 3.0,
            "adx_period": 14,
            "adx_threshold": 20,
            "sma_regime_period": 200,
            "risk_pct": 0.01,
            "max_pct": 0.2,
            "stop_loss_pct": 0.015,
            "max_drawdown": 0.15
        }
        
        if parameters:
            default_parameters.update(parameters)
        
        super().__init__(default_parameters)
        
        self.name = "tech_momentum_ema_cross"
        self.version = "1.0.0"
        self.description = "Momentum trend-following strategy with dual EMA cross and MACD/RSI confirmation"
        self.symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]
        self.strategy_type = "momentum"
        
        # Track previous values for crossover detection
        self._prev_price = None
        self._prev_ema_fast = None
        self._prev_macd_hist = None
        
        logger.info(f"Initialized {self.name} v{self.version}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on EMA crossover with RSI and MACD confirmation.
        
        Args:
            data: DataFrame with OHLCV and indicator columns. Last row is most recent.
        
        Returns:
            pd.Series with Signal enum values (BUY, SELL, HOLD, CLOSE)
        """
        signals = pd.Series(index=data.index, data=Signal.HOLD)
        
        if len(data) < 2:
            logger.warning("Insufficient data for signal generation")
            return signals
        
        # Extract parameters
        ema_fast_period = self.parameters.get("ema_fast_period", 9)
        ema_slow_period = self.parameters.get("ema_slow_period", 21)
        rsi_period = self.parameters.get("rsi_period", 14)
        rsi_min = self.parameters.get("rsi_entry_min", 50)
        rsi_max = self.parameters.get("rsi_entry_max", 70)
        rsi_exit = self.parameters.get("rsi_exit_threshold", 75)
        
        # Column name mappings for ATLAS indicators
        ema_fast_col = f"ema_{ema_fast_period}"
        ema_slow_col = f"ema_{ema_slow_period}"
        rsi_col = f"rsi_{rsi_period}"
        macd_hist_col = "macd_histogram"
        macd_line_col = "macd_line"
        macd_signal_col = "macd_signal"
        atr_col = f"atr_{self.parameters.get('atr_period', 14)}"
        
        # Iterate through data to detect signals
        for i in range(1, len(data)):
            idx = data.index[i]
            prev_idx = data.index[i - 1]
            
            current_row = data.iloc[i]
            prev_row = data.iloc[i - 1]
            
            # Check for required columns and valid values
            close = current_row.get("close", np.nan)
            prev_close = prev_row.get("close", np.nan)
            
            if pd.isna(close) or pd.isna(prev_close):
                continue
            
            # Get EMA values
            ema_fast = current_row.get(ema_fast_col, np.nan)
            ema_slow = current_row.get(ema_slow_col, np.nan)
            prev_ema_fast = prev_row.get(ema_fast_col, np.nan)
            prev_ema_slow = prev_row.get(ema_slow_col, np.nan)
            
            # Get RSI
            rsi = current_row.get(rsi_col, np.nan)
            
            # Get MACD histogram
            macd_hist = current_row.get(macd_hist_col, np.nan)
            prev_macd_hist = prev_row.get(macd_hist_col, np.nan)
            
            # Check if we have all required indicators
            if any(pd.isna(x) for x in [ema_fast, ema_slow, rsi, macd_hist]):
                continue
            
            # ENTRY SIGNAL: Check all conditions
            # 1. Price crosses above EMA-9 (fast EMA)
            price_crossed_above_ema9 = prev_close <= prev_ema_fast and close > ema_fast
            
            # 2. Price is above EMA-21 (slow EMA)
            price_above_ema21 = close > ema_slow
            
            # 3. RSI is in momentum zone (50-70)
            rsi_in_zone = rsi_min <= rsi <= rsi_max
            
            # 4. MACD histogram turns positive
            macd_turned_positive = prev_macd_hist <= 0 and macd_hist > 0
            
            # 5. Check regime filter (done in check_filters method, but also check here)
            regime_ok = self.check_filters(data.iloc[:i+1])
            
            if (price_crossed_above_ema9 and 
                price_above_ema21 and 
                rsi_in_zone and 
                macd_turned_positive and 
                regime_ok):
                signals.iloc[i] = Signal.BUY
                logger.info(f"BUY signal at {idx}: price={close:.2f}, EMA9={ema_fast:.2f}, "
                           f"EMA21={ema_slow:.2f}, RSI={rsi:.2f}, MACD_hist={macd_hist:.4f}")
            
            # EXIT SIGNAL: Check exit conditions
            # 1. RSI overbought (>75)
            rsi_overbought = rsi > rsi_exit
            
            # 2. Price crosses below EMA-21
            price_crossed_below_ema21 = prev_close >= prev_ema_slow and close < ema_slow
            
            if rsi_overbought or price_crossed_below_ema21:
                signals.iloc[i] = Signal.CLOSE
                logger.info(f"CLOSE signal at {idx}: RSI={rsi:.2f}, price={close:.2f}, "
                           f"EMA21={ema_slow:.2f}")
        
        return signals
    
    def compute_position_size(self, signal, portfolio_value: float) -> float:
        """
        Compute position size based on fixed fractional risk management.
        
        Args:
            signal: Signal enum or TradeSignal object
            portfolio_value: Current portfolio value in dollars
        
        Returns:
            Number of shares/units to trade
        """
        if portfolio_value <= 0:
            logger.warning("Portfolio value is zero or negative")
            return 0.0
        
        # Extract parameters
        risk_pct = self.parameters.get("risk_pct", 0.01)
        max_pct = self.parameters.get("max_pct", 0.2)
        
        # Calculate risk amount (1% of portfolio by default)
        risk_amount = portfolio_value * risk_pct
        
        # Calculate max position size (20% of portfolio by default)
        max_position_value = portfolio_value * max_pct
        
        # Get current price and ATR for stop calculation
        if isinstance(signal, TradeSignal):
            price = signal.price
            atr = getattr(signal, 'atr', None)
        else:
            # If we only have a Signal enum, we need more context
            # Return a conservative position based on max_pct only
            logger.warning("Position sizing without full signal context")
            return max_position_value / 100.0  # Assume $100 per share as placeholder
        
        if price is None or price <= 0:
            logger.warning("Invalid price for position sizing")
            return 0.0
        
        # Calculate stop distance
        atr_multiple = self.parameters.get("atr_stop_multiple", 1.5)
        
        if atr and atr > 0:
            stop_distance = atr * atr_multiple
        else:
            # Fallback to percentage stop
            stop_distance = price * self.parameters.get("stop_loss_pct", 0.015)
        
        # Position size based on risk: risk_amount / stop_distance
        if stop_distance > 0:
            position_size_risk = risk_amount / stop_distance
        else:
            position_size_risk = 0.0
        
        # Position size based on max allocation
        position_size_max = max_position_value / price
        
        # Take the minimum of the two to respect both constraints
        position_size = min(position_size_risk, position_size_max)
        
        # Round down to whole shares
        position_size = int(np.floor(position_size))
        
        logger.info(f"Position size: {position_size} shares @ ${price:.2f} "
                   f"(risk-based: {position_size_risk:.2f}, max: {position_size_max:.2f})")
        
        return float(position_size)
    
    def check_filters(self, data: pd.DataFrame) -> bool:
        """
        Check if regime and market filters pass for trading.
        
        Args:
            data: DataFrame with indicator columns. Last row is most recent.
        
        Returns:
            True if filters pass and trading is allowed, False otherwise
        """
        if len(data) == 0:
            logger.warning("Empty data for filter check")
            return False
        
        current_row = data.iloc[-1]
        
        # Extract parameters
        sma_period = self.parameters.get("sma_regime_period", 200)
        adx_period = self.parameters.get("adx_period", 14)
        adx_threshold = self.parameters.get("adx_threshold", 20)
        
        # Check required columns
        close = current_row.get("close", np.nan)
        sma_col = f"sma_{sma_period}"
        adx_col = f"adx_{adx_period}"
        
        sma = current_row.get(sma_col, np.nan)
        adx = current_row.get(adx_col, np.nan)
        
        # 1. Regime Filter: Close must be above SMA-200
        if pd.isna(sma) or pd.isna(close):
            logger.warning(f"Missing regime filter data: close={close}, SMA-{sma_period}={sma}")
            return False
        
        if close <= sma:
            logger.debug(f"Regime filter failed: close ({close:.2f}) <= SMA-{sma_period} ({sma:.2f})")
            return False
        
        # 2. Trend Filter: ADX must be above threshold (trending market)
        if pd.isna(adx):
            logger.warning(f"Missing ADX-{adx_period} data")
            return False
        
        if adx < adx_threshold:
            logger.debug(f"ADX filter failed: ADX ({adx:.2f}) < threshold ({adx_threshold})")
            return False
        
        logger.debug(f"Filters passed: close={close:.2f} > SMA={sma:.2f}, ADX={adx:.2f} > {adx_threshold}")
        return True
    
    def get_metadata(self) -> dict:
        """
        Return strategy metadata.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "strategy_type": self.strategy_type,
            "symbols": self.symbols,
            "parameters": self.parameters,
            "timeframe": "1h",
            "asset_class": "equity",
            "entry_rules": {
                "price_above_ema9": "Close price crosses above EMA-9",
                "price_above_ema21": "Close price is above EMA-21",
                "rsi_momentum_zone": f"RSI-{self.parameters.get('rsi_period', 14)} is between {self.parameters.get('rsi_entry_min', 50)} and {self.parameters.get('rsi_entry_max', 70)}",
                "regime_confirmation": f"Daily close price is above SMA-{self.parameters.get('sma_regime_period', 200)}",
                "macd_histogram_positive": "MACD histogram turns positive"
            },
            "exit_rules": {
                "stop_loss_hit": f"Price falls below entry minus {self.parameters.get('atr_stop_multiple', 1.5)}× ATR-{self.parameters.get('atr_period', 14)}",
                "rsi_overbought": f"RSI-{self.parameters.get('rsi_period', 14)} exceeds {self.parameters.get('rsi_exit_threshold', 75)}",
                "take_profit_hit": f"Price reaches entry plus {self.parameters.get('atr_target_multiple', 3.0)}× ATR-{self.parameters.get('atr_period', 14)}",
                "price_below_ema21": "Close price crosses below EMA-21"
            },
            "regime_filter": {
                "required_trend": "bullish",
                "min_adx": self.parameters.get("adx_threshold", 20),
                "sma_period": self.parameters.get("sma_regime_period", 200)
            }
        }