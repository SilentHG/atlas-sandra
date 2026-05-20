from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
from loguru import logger
from strategy_engine.base_strategy import BaseStrategy, TradeSignal, Signal


class TechMomentumEmaCrossover(BaseStrategy):
    """
    A momentum trend-following strategy for large-cap US tech equities that enters on EMA 
    crossovers with RSI confirmation and MACD histogram support. Exits are triggered by 
    overbought RSI conditions or EMA breakdown. Trades only in confirmed bullish daily 
    regimes above the 200-day moving average.
    
    Entry Conditions:
    - Price crosses above both EMA-9 and EMA-21 or is already above both EMAs
    - RSI-14 is between 50 and 70 inclusive
    - MACD histogram turns positive or is positive and rising
    - Volume is at least 1.2x the 20-period average volume
    
    Exit Conditions:
    - RSI-14 rises above 75 (overbought)
    - Price closes below EMA-21 (breakdown)
    - Stop loss hit (entry - 1.5x ATR-14)
    - Position held for more than 48 hours without profit target and RSI declining
    
    Regime Filter:
    - Daily price above SMA-200 (bullish trend)
    - ADX-14 above 20 (trend strength)
    - VIX below 30 (avoid extreme volatility)
    """
    
    def __init__(self, parameters: dict = None):
        """Initialize the strategy with given parameters."""
        super().__init__(parameters)
        self.name = "tech_momentum_ema_crossover"
        self.version = "1.0.0"
        self.description = "EMA crossover momentum strategy with RSI/MACD confirmation for tech stocks"
        self.strategy_type = "momentum"
        self.symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]
        
        # Set default parameters if not provided
        if self.parameters is None:
            self.parameters = {}
        
        # EMA parameters
        self.parameters.setdefault('ema_fast', 9)
        self.parameters.setdefault('ema_slow', 21)
        
        # RSI parameters
        self.parameters.setdefault('rsi_period', 14)
        self.parameters.setdefault('rsi_entry_min', 50)
        self.parameters.setdefault('rsi_entry_max', 70)
        self.parameters.setdefault('rsi_exit_threshold', 75)
        
        # MACD parameters
        self.parameters.setdefault('macd_fast', 12)
        self.parameters.setdefault('macd_slow', 26)
        self.parameters.setdefault('macd_signal', 9)
        
        # Volume parameters
        self.parameters.setdefault('volume_ma_period', 20)
        self.parameters.setdefault('volume_multiplier', 1.2)
        
        # ATR parameters
        self.parameters.setdefault('atr_period', 14)
        self.parameters.setdefault('atr_stop_multiple', 1.5)
        
        # Regime filter parameters
        self.parameters.setdefault('sma_regime', 200)
        self.parameters.setdefault('adx_period', 14)
        self.parameters.setdefault('adx_threshold', 20)
        self.parameters.setdefault('max_vix', 30)
        
        # Position sizing parameters
        self.parameters.setdefault('risk_pct', 0.01)
        self.parameters.setdefault('max_pct', 0.15)
        
        # Exit parameters
        self.parameters.setdefault('max_holding_hours', 48)
        self.parameters.setdefault('target_rr', 2.5)
        
        # Track position entry details for time-based exits
        self.position_entry_time = None
        self.position_entry_price = None
        self.position_rsi_peak = None
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on EMA crossover, RSI, MACD histogram, and volume.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators. Latest bar is last row.
            
        Returns:
            pd.Series with Signal enum values: BUY, SELL, HOLD, or CLOSE
        """
        if data.empty or len(data) < 2:
            logger.warning("Insufficient data for signal generation")
            return pd.Series([Signal.HOLD] * len(data), index=data.index)
        
        signals = pd.Series(Signal.HOLD, index=data.index)
        
        # Extract parameters
        ema_fast = self.parameters['ema_fast']
        ema_slow = self.parameters['ema_slow']
        rsi_period = self.parameters['rsi_period']
        rsi_min = self.parameters['rsi_entry_min']
        rsi_max = self.parameters['rsi_entry_max']
        rsi_exit = self.parameters['rsi_exit_threshold']
        volume_multiplier = self.parameters['volume_multiplier']
        volume_ma_period = self.parameters['volume_ma_period']
        
        # Build column names for indicators
        ema_fast_col = f'ema_{ema_fast}'
        ema_slow_col = f'ema_{ema_slow}'
        rsi_col = f'rsi_{rsi_period}'
        macd_hist_col = 'macd_histogram'
        volume_ma_col = f'volume_sma_{volume_ma_period}'
        
        # Check if required columns exist
        required_cols = [ema_fast_col, ema_slow_col, rsi_col, macd_hist_col, 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"Required column {col} not found in data")
                return signals
        
        # Iterate through data to generate signals
        for i in range(1, len(data)):
            idx = data.index[i]
            prev_idx = data.index[i-1]
            
            # Get current and previous bar data
            current_close = data.loc[idx, 'close']
            current_volume = data.loc[idx, 'volume']
            current_rsi = data.loc[idx, rsi_col]
            current_macd_hist = data.loc[idx, macd_hist_col]
            current_ema_fast = data.loc[idx, ema_fast_col]
            current_ema_slow = data.loc[idx, ema_slow_col]
            
            prev_macd_hist = data.loc[prev_idx, macd_hist_col]
            
            # Check for NaN values
            if pd.isna(current_close) or pd.isna(current_rsi) or pd.isna(current_macd_hist):
                continue
            if pd.isna(current_ema_fast) or pd.isna(current_ema_slow):
                continue
            
            # Check volume MA if column exists
            volume_check = True
            if volume_ma_col in data.columns and not pd.isna(data.loc[idx, volume_ma_col]):
                volume_ma = data.loc[idx, volume_ma_col]
                volume_check = current_volume >= (volume_multiplier * volume_ma)
            
            # Entry Logic: BUY signal
            # 1. Price above both EMAs
            price_above_emas = current_close > current_ema_fast and current_close > current_ema_slow
            
            # 2. RSI between 50 and 70
            rsi_in_range = rsi_min <= current_rsi <= rsi_max
            
            # 3. MACD histogram turns positive or is positive and rising
            macd_positive_turn = (current_macd_hist > 0 and prev_macd_hist <= 0)
            macd_positive_rising = (current_macd_hist > 0 and current_macd_hist > prev_macd_hist)
            macd_check = macd_positive_turn or macd_positive_rising
            
            # 4. Volume confirmation
            # Combined entry condition
            if price_above_emas and rsi_in_range and macd_check and volume_check:
                signals.iloc[i] = Signal.BUY
                logger.info(f"BUY signal at {idx}: RSI={current_rsi:.2f}, MACD_hist={current_macd_hist:.4f}")
            
            # Exit Logic: SELL/CLOSE signal
            # 1. RSI overbought (above 75)
            rsi_overbought = current_rsi > rsi_exit
            
            # 2. Price closes below EMA-21 (breakdown)
            ema_breakdown = current_close < current_ema_slow
            
            if rsi_overbought or ema_breakdown:
                signals.iloc[i] = Signal.CLOSE
                reason = "RSI overbought" if rsi_overbought else "EMA breakdown"
                logger.info(f"CLOSE signal at {idx}: {reason}, RSI={current_rsi:.2f}, Close={current_close:.2f}, EMA21={current_ema_slow:.2f}")
        
        return signals
    
    def compute_position_size(self, signal, portfolio_value: float) -> float:
        """
        Compute position size based on fixed fractional risk method.
        
        Args:
            signal: Signal enum or TradeSignal object
            portfolio_value: Current portfolio value in dollars
            
        Returns:
            Number of shares/units to trade
        """
        if portfolio_value <= 0:
            logger.warning("Portfolio value must be positive")
            return 0.0
        
        # Extract risk parameters
        risk_pct = self.parameters.get('risk_pct', 0.01)
        max_pct = self.parameters.get('max_pct', 0.15)
        
        # Maximum position size based on max_pct
        max_position_value = portfolio_value * max_pct
        
        # Risk-based position size
        # Risk amount = portfolio_value * risk_pct
        risk_amount = portfolio_value * risk_pct
        
        # For stop loss calculation, we need entry price and ATR
        # Since we don't have entry price here, we use a conservative approach
        # Assume stop loss is atr_multiple * ATR
        atr_multiple = self.parameters.get('atr_stop_multiple', 1.5)
        
        # Extract current price from signal if available
        if isinstance(signal, TradeSignal):
            entry_price = getattr(signal, 'entry_price', None)
            if entry_price and entry_price > 0:
                # Position size based on risk: risk_amount / stop_loss_distance
                stop_loss_distance = entry_price * atr_multiple * 0.01  # Rough approximation
                if stop_loss_distance > 0:
                    risk_based_shares = risk_amount / stop_loss_distance
                    risk_based_value = risk_based_shares * entry_price
                    
                    # Take minimum of risk-based and max position size
                    position_value = min(risk_based_value, max_position_value)
                    shares = position_value / entry_price
                    
                    logger.info(f"Position size: {shares:.2f} shares (${position_value:.2f})")
                    return shares
        
        # Fallback: use max_pct of portfolio
        # Assume we need a price; without it, return 0
        logger.warning("Unable to compute position size without entry price")
        return 0.0
    
    def check_filters(self, data: pd.DataFrame) -> bool:
        """
        Check regime and market filters to determine if trading is allowed.
        
        Args:
            data: DataFrame with price data and indicators
            
        Returns:
            True if all filters pass and trading is allowed, False otherwise
        """
        if data.empty:
            logger.warning("Empty data for filter check")
            return False
        
        # Get latest bar (most recent)
        latest = data.iloc[-1]
        
        # Extract regime filter parameters
        sma_regime = self.parameters.get('sma_regime', 200)
        adx_threshold = self.parameters.get('adx_threshold', 20)
        max_vix = self.parameters.get('max_vix', 30)
        
        # Build column names
        sma_col = f'sma_{sma_regime}'
        adx_col = f'adx_{self.parameters.get("adx_period", 14)}'
        
        # Check 1: Price above SMA-200 (bullish regime)
        if sma_col not in data.columns or pd.isna(latest[sma_col]):
            logger.warning(f"SMA-{sma_regime} not available for regime filter")
            return False
        
        if 'close' not in data.columns or pd.isna(latest['close']):
            logger.warning("Close price not available for regime filter")
            return False
        
        price_above_sma = latest['close'] > latest[sma_col]
        if not price_above_sma:
            logger.info(f"Regime filter FAIL: Price {latest['close']:.2f} below SMA-{sma_regime} {latest[sma_col]:.2f}")
            return False
        
        # Check 2: ADX above threshold (trend strength)
        if adx_col not in data.columns or pd.isna(latest[adx_col]):
            logger.warning(f"ADX-{self.parameters.get('adx_period', 14)} not available for regime filter")
            return False
        
        adx_above_threshold = latest[adx_col] > adx_threshold
        if not adx_above_threshold:
            logger.info(f"Regime filter FAIL: ADX {latest[adx_col]:.2f} below threshold {adx_threshold}")
            return False
        
        # Check 3: VIX below max (avoid extreme volatility)
        # VIX might be available as 'vix' or '^VIX' column
        vix_check = True
        if 'vix' in data.columns and not pd.isna(latest['vix']):
            vix_check = latest['vix'] < max_vix
            if not vix_check:
                logger.info(f"Regime filter FAIL: VIX {latest['vix']:.2f} above max {max_vix}")
                return False
        elif '^VIX' in data.columns and not pd.isna(latest['^VIX']):
            vix_check = latest['^VIX'] < max_vix
            if not vix_check:
                logger.info(f"Regime filter FAIL: VIX {latest['^VIX']:.2f} above max {max_vix}")
                return False
        
        logger.info("All regime filters PASS")
        return True
    
    def get_metadata(self) -> dict:
        """
        Return strategy metadata including name, version, description, type, symbols, and parameters.
        
        Returns:
            Dictionary containing strategy metadata
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'strategy_type': self.strategy_type,
            'symbols': self.symbols,
            'parameters': self.parameters,
            'timeframe': '1h',
            'asset_class': 'equity',
            'entry_rules': {
                'ema_crossover': 'Price crosses above both EMA-9 and EMA-21 or is above both EMAs',
                'rsi_momentum': 'RSI-14 is between 50 and 70 inclusive',
                'macd_confirmation': 'MACD histogram turns positive or is positive and rising',
                'volume_confirmation': 'Current bar volume is at least 1.2x the 20-period average volume'
            },
            'exit_rules': {
                'rsi_overbought': 'RSI-14 rises above 75',
                'ema_breakdown': 'Price closes below EMA-21',
                'stop_loss_hit': 'Price falls below entry price minus 1.5x ATR-14',
                'time_stop': 'Position held for more than 48 hours without hitting profit target and RSI declining from peak'
            },
            'regime_filter': {
                'required_trend': 'bullish',
                'min_adx': self.parameters.get('adx_threshold', 20),
                'max_vix': self.parameters.get('max_vix', 30),
                'description': 'Daily price must be above SMA-200 and ADX-14 above 20 indicating established trend strength. VIX below 30 to avoid extreme volatility periods.'
            }
        }