from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
from loguru import logger
from strategy_engine.base_strategy import BaseStrategy, TradeSignal, Signal


"""
Tech Momentum EMA Crossover Strategy

A momentum trend-following strategy for large-cap US tech stocks that enters on EMA 
crossovers with RSI confirmation and MACD histogram support. Exits on overbought 
conditions or trend reversal signals. Trades only in bullish daily regime conditions.

Entry Conditions:
- Close price crosses above EMA-9
- Close price is above EMA-21
- RSI-14 is between 50 and 70 (momentum zone)
- MACD histogram is positive
- Daily regime: price above SMA-200 and ADX > 20

Exit Conditions:
- RSI-14 exceeds 75 (overbought)
- Price crosses below EMA-21 (trend reversal)
- Stop loss: Entry price minus 1.5x ATR-14
- End of day: 15 minutes before close

Position Sizing: Risk-based using 1% account risk per trade, max 20% portfolio
"""


class TechMomentumEmaCrossover(BaseStrategy):
    """
    Tech Momentum EMA Crossover strategy implementation.
    
    This strategy trades large-cap tech stocks (AAPL, MSFT, NVDA, GOOGL, META) on 
    1-hour timeframe, entering on EMA crossovers with multi-indicator confirmation 
    and exiting on overbought signals or trend breaks.
    """
    
    def __init__(self, parameters: dict = None):
        """
        Initialize the Tech Momentum EMA Crossover strategy.
        
        Args:
            parameters: Strategy configuration parameters
        """
        default_params = {
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
            "regime_sma_period": 200,
            "adx_period": 14,
            "adx_threshold": 20,
            "risk_pct": 0.01,
            "max_pct": 0.2,
            "stop_loss_pct": 0.02
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(default_params)
        
        # Track previous values for crossover detection
        self.prev_close = None
        self.prev_ema_fast = None
        self.prev_ema_slow = None
        
        logger.info(f"Initialized TechMomentumEmaCrossover with parameters: {self.parameters}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on EMA crossover with RSI and MACD confirmation.
        
        The most recent bar is the LAST row in the data DataFrame.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            pd.Series with Signal enum values (BUY, SELL, HOLD, CLOSE)
        """
        signals = pd.Series(Signal.HOLD, index=data.index)
        
        if len(data) < 2:
            logger.warning("Insufficient data for signal generation")
            return signals
        
        # Extract parameters
        ema_fast_period = self.parameters.get("ema_fast_period", 9)
        ema_slow_period = self.parameters.get("ema_slow_period", 21)
        rsi_period = self.parameters.get("rsi_period", 14)
        rsi_entry_min = self.parameters.get("rsi_entry_min", 50)
        rsi_entry_max = self.parameters.get("rsi_entry_max", 70)
        rsi_exit_threshold = self.parameters.get("rsi_exit_threshold", 75)
        
        # Column names for indicators (adjust based on ATLAS feature names)
        close_col = 'close'
        ema_fast_col = f'ema_{ema_fast_period}'
        ema_slow_col = f'ema_{ema_slow_period}'
        rsi_col = f'rsi_{rsi_period}'
        macd_hist_col = 'macd_histogram'
        
        # Check if required columns exist
        required_cols = [close_col, ema_fast_col, ema_slow_col, rsi_col, macd_hist_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return signals
        
        # Iterate through data to generate signals
        for i in range(1, len(data)):
            current_idx = data.index[i]
            prev_idx = data.index[i - 1]
            
            # Current values
            close = data.loc[current_idx, close_col]
            ema_fast = data.loc[current_idx, ema_fast_col]
            ema_slow = data.loc[current_idx, ema_slow_col]
            rsi = data.loc[current_idx, rsi_col]
            macd_hist = data.loc[current_idx, macd_hist_col]
            
            # Previous values
            prev_close = data.loc[prev_idx, close_col]
            prev_ema_fast = data.loc[prev_idx, ema_fast_col]
            
            # Check for NaN values
            if pd.isna(close) or pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(rsi) or pd.isna(macd_hist):
                signals.loc[current_idx] = Signal.HOLD
                continue
                
            if pd.isna(prev_close) or pd.isna(prev_ema_fast):
                signals.loc[current_idx] = Signal.HOLD
                continue
            
            # Entry Signal: Close crosses above EMA-9
            price_crossed_above_ema9 = (prev_close <= prev_ema_fast) and (close > ema_fast)
            
            # Additional entry conditions
            price_above_ema21 = close > ema_slow
            rsi_in_momentum_zone = rsi_entry_min <= rsi <= rsi_entry_max
            macd_positive = macd_hist > 0
            
            # Check all entry conditions
            if price_crossed_above_ema9 and price_above_ema21 and rsi_in_momentum_zone and macd_positive:
                # Verify regime filter before signaling BUY
                if self.check_filters(data.iloc[:i+1]):
                    signals.loc[current_idx] = Signal.BUY
                    logger.info(f"BUY signal at {current_idx}: close={close:.2f}, ema9={ema_fast:.2f}, "
                              f"ema21={ema_slow:.2f}, rsi={rsi:.2f}, macd_hist={macd_hist:.4f}")
            
            # Exit Signal: RSI overbought
            elif rsi > rsi_exit_threshold:
                signals.loc[current_idx] = Signal.CLOSE
                logger.info(f"CLOSE signal (RSI overbought) at {current_idx}: rsi={rsi:.2f}")
            
            # Exit Signal: Price crosses below EMA-21 (trend reversal)
            elif close < ema_slow and prev_close >= data.loc[prev_idx, ema_slow_col]:
                signals.loc[current_idx] = Signal.CLOSE
                logger.info(f"CLOSE signal (trend reversal) at {current_idx}: close={close:.2f}, ema21={ema_slow:.2f}")
            
            else:
                signals.loc[current_idx] = Signal.HOLD
        
        return signals
    
    def compute_position_size(self, signal, portfolio_value: float) -> float:
        """
        Compute position size based on risk-based method.
        
        Uses fixed fractional risk: risk 1% of portfolio per trade, with max 20% position size.
        Position size = (Portfolio * RiskPct) / (StopLoss Distance)
        
        Args:
            signal: Signal enum or TradeSignal object
            portfolio_value: Current portfolio value in dollars
            
        Returns:
            Number of shares to trade (float)
        """
        if portfolio_value <= 0:
            logger.warning(f"Invalid portfolio value: {portfolio_value}")
            return 0.0
        
        risk_pct = self.parameters.get("risk_pct", 0.01)
        max_pct = self.parameters.get("max_pct", 0.2)
        atr_stop_multiple = self.parameters.get("atr_stop_multiple", 1.5)
        
        # Extract signal information
        if isinstance(signal, TradeSignal):
            entry_price = signal.price
            stop_loss_price = signal.stop_loss
        else:
            # If only Signal enum provided, we need current market data
            # Use a default stop loss percentage
            logger.warning("Signal object not provided, using default stop loss calculation")
            return 0.0
        
        if entry_price is None or entry_price <= 0:
            logger.warning("Invalid entry price for position sizing")
            return 0.0
        
        # Calculate stop loss distance
        if stop_loss_price and stop_loss_price > 0:
            stop_distance = entry_price - stop_loss_price
        else:
            # Fallback to percentage-based stop
            stop_loss_pct = self.parameters.get("stop_loss_pct", 0.02)
            stop_distance = entry_price * stop_loss_pct
        
        if stop_distance <= 0:
            logger.warning(f"Invalid stop distance: {stop_distance}")
            return 0.0
        
        # Calculate risk-based position size
        risk_amount = portfolio_value * risk_pct
        shares = risk_amount / stop_distance
        
        # Apply maximum position size constraint
        max_position_value = portfolio_value * max_pct
        max_shares = max_position_value / entry_price
        
        final_shares = min(shares, max_shares)
        
        logger.debug(f"Position sizing: portfolio={portfolio_value:.2f}, risk_amount={risk_amount:.2f}, "
                    f"stop_distance={stop_distance:.2f}, shares={shares:.2f}, max_shares={max_shares:.2f}, "
                    f"final_shares={final_shares:.2f}")
        
        return final_shares
    
    def check_filters(self, data: pd.DataFrame) -> bool:
        """
        Check if regime and market filters pass for trading.
        
        Regime Filter:
        - Daily close price must be above SMA-200 (bullish regime)
        - ADX-14 must be above 20 (trending conditions)
        
        Args:
            data: DataFrame with market data and indicators
            
        Returns:
            True if all filters pass, False otherwise
        """
        if len(data) == 0:
            logger.warning("Empty data for filter check")
            return False
        
        # Get parameters
        regime_sma_period = self.parameters.get("regime_sma_period", 200)
        adx_period = self.parameters.get("adx_period", 14)
        adx_threshold = self.parameters.get("adx_threshold", 20)
        
        # Use the most recent bar (last row)
        current_bar = data.iloc[-1]
        
        # Column names
        close_col = 'close'
        sma_col = f'sma_{regime_sma_period}'
        adx_col = f'adx_{adx_period}'
        
        # Check for required columns
        if close_col not in data.columns:
            logger.warning(f"Missing {close_col} column for regime filter")
            return False
        
        if sma_col not in data.columns:
            logger.warning(f"Missing {sma_col} column for regime filter")
            return False
        
        if adx_col not in data.columns:
            logger.warning(f"Missing {adx_col} column for regime filter")
            return False
        
        close = current_bar[close_col]
        sma = current_bar[sma_col]
        adx = current_bar[adx_col]
        
        # Check for NaN values
        if pd.isna(close) or pd.isna(sma) or pd.isna(adx):
            logger.warning(f"NaN values in filter check: close={close}, sma={sma}, adx={adx}")
            return False
        
        # Check bullish regime: price above SMA-200
        bullish_regime = close > sma
        
        # Check trending conditions: ADX above threshold
        trending = adx > adx_threshold
        
        filters_pass = bullish_regime and trending
        
        if not filters_pass:
            logger.debug(f"Filters failed: bullish_regime={bullish_regime} (close={close:.2f}, sma={sma:.2f}), "
                        f"trending={trending} (adx={adx:.2f}, threshold={adx_threshold})")
        
        return filters_pass
    
    def get_metadata(self) -> dict:
        """
        Get strategy metadata and configuration.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            "name": "tech_momentum_ema_crossover",
            "version": "1.0.0",
            "description": "A momentum trend-following strategy for large-cap US tech stocks that enters "
                         "on EMA crossovers with RSI confirmation and MACD histogram support. Exits on "
                         "overbought conditions or trend reversal signals. Trades only in bullish daily "
                         "regime conditions.",
            "strategy_type": "momentum",
            "symbols": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
            "timeframe": "1h",
            "parameters": self.parameters,
            "entry_rules": {
                "price_above_ema9": "Close price crosses above EMA-9",
                "price_above_ema21": "Close price is above EMA-21",
                "rsi_momentum_zone": "RSI-14 is between 50 and 70 (inclusive)",
                "macd_confirmation": "MACD histogram is positive (above zero line)",
                "regime_alignment": "Daily close price is above SMA-200 (bullish regime)"
            },
            "exit_rules": {
                "rsi_overbought": "RSI-14 exceeds 75",
                "trend_reversal": "Close price crosses below EMA-21",
                "stop_loss_hit": "Price falls below entry price minus 1.5× ATR-14",
                "end_of_day": "Exit all positions 15 minutes before market close if still open"
            },
            "regime_filter": {
                "required_trend": "bullish",
                "min_adx": 20,
                "description": "Trade only when daily price is above SMA-200 and ADX-14 indicates trending conditions (ADX > 20)"
            }
        }
    
    def create_trade_signal(self, signal_type: Signal, data: pd.DataFrame) -> TradeSignal:
        """
        Create a TradeSignal object with stop loss and take profit levels.
        
        Args:
            signal_type: The type of signal (BUY, SELL, CLOSE, HOLD)
            data: DataFrame with current market data
            
        Returns:
            TradeSignal object with calculated levels
        """
        if len(data) == 0:
            logger.warning("Empty data for creating trade signal")
            return TradeSignal(
                signal=signal_type,
                timestamp=pd.Timestamp.now(),
                price=0.0,
                confidence=0.0
            )
        
        current_bar = data.iloc[-1]
        close_col = 'close'
        atr_period = self.parameters.get("atr_period", 14)
        atr_col = f'atr_{atr_period}'
        
        if close_col not in data.columns:
            logger.warning(f"Missing {close_col} column")
            return TradeSignal(
                signal=signal_type,
                timestamp=pd.Timestamp.now(),
                price=0.0,
                confidence=0.0
            )
        
        price = current_bar[close_col]
        timestamp = current_bar.name if hasattr(current_bar, 'name') else pd.Timestamp.now()
        
        # Calculate stop loss and take profit for BUY signals
        stop_loss = None
        take_profit = None
        
        if signal_type == Signal.BUY:
            # Calculate ATR-based stop loss
            if atr_col in data.columns and not pd.isna(current_bar[atr_col]):
                atr = current_bar[atr_col]
                atr_multiple = self.parameters.get("atr_stop_multiple", 1.5)
                stop_loss = price - (atr * atr_multiple)
                
                # Take profit is dynamic (not fixed), triggered by RSI overbought or trend break
                # Set a nominal take profit at 2x risk for tracking purposes
                risk = price - stop_loss
                take_profit = price + (risk * 2.0)
            else:
                # Fallback to percentage-based stop loss
                stop_loss_pct = self.parameters.get("stop_loss_pct", 0.02)
                stop_loss = price * (1 - stop_loss_pct)
                take_profit = price * (1 + stop_loss_pct * 2)
        
        # Calculate confidence based on multiple indicator alignment
        confidence = self._calculate_confidence(data)
        
        return TradeSignal(
            signal=signal_type,
            timestamp=timestamp,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            metadata={
                "strategy": "tech_momentum_ema_crossover",
                "trade_id": str(uuid.uuid4())
            }
        )
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        """
        Calculate confidence score based on multiple indicator alignment.
        
        Args:
            data: DataFrame with market data and indicators
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if len(data) == 0:
            return 0.0
        
        current_bar = data.iloc[-1]
        confidence_factors = []
        
        # Factor 1: RSI in optimal momentum zone (50-60 is ideal)
        rsi_period = self.parameters.get("rsi_period", 14)
        rsi_col = f'rsi_{rsi_period}'
        if rsi_col in data.columns and not pd.isna(current_bar[rsi_col]):
            rsi = current_bar[rsi_col]
            # Highest confidence at RSI = 55, decreasing towards edges
            if 50 <= rsi <= 60:
                confidence_factors.append(1.0)
            elif 60 < rsi <= 70:
                confidence_factors.append(0.8)
            elif 45 < rsi < 50:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
        
        # Factor 2: MACD histogram strength
        macd_hist_col = 'macd_histogram'
        if macd_hist_col in data.columns and not pd.isna(current_bar[macd_hist_col]):
            macd_hist = current_bar[macd_hist_col]
            if macd_hist > 0.5:
                confidence_factors.append(1.0)
            elif macd_hist > 0.2:
                confidence_factors.append(0.8)
            elif macd_hist > 0:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.2)
        
        # Factor 3: ADX strength (higher ADX = stronger trend)
        adx_period = self.parameters.get("adx_period", 14)
        adx_col = f'adx_{adx_period}'
        adx_threshold = self.parameters.get("adx_threshold", 20)
        if adx_col in data.columns and not pd.isna(current_bar[adx_col]):
            adx = current_bar[adx_col]
            if adx > 30:
                confidence_factors.append(1.0)
            elif adx > 25:
                confidence_factors.append(0.8)
            elif adx > adx_threshold:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
        
        # Factor 4: Distance from EMA-21 (closer = better entry timing)
        ema_slow_period = self.parameters.get("ema_slow_period", 21)
        ema_slow_col = f'ema_{ema_slow_period}'
        close_col = 'close'
        if ema_slow_col in data.columns and close_col in data.columns:
            if not pd.isna(current_bar[ema_slow_col]) and not pd.isna(current_bar[close_col]):
                close = current_bar[close_col]
                ema_slow = current_bar[ema_slow_col]
                distance_pct = ((close - ema_slow) / ema_slow) * 100
                if 0 < distance_pct <= 1:
                    confidence_factors.append(1.0)
                elif 1 < distance_pct <= 2:
                    confidence_factors.append(0.8)
                elif 2 < distance_pct <= 3:
                    confidence_factors.append(0.6)
                else:
                    confidence_factors.append(0.5)
        
        # Calculate average confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5