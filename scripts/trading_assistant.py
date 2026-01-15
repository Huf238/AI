"""
============================================================================
Trading Assistant - Real-Time Stock Analysis with AI
============================================================================
This script fetches real-time stock data for any ticker and uses your
trained AI model to analyze it and suggest entries.

USAGE:
    # Interactive mode
    python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_*
    
    # Quick analysis
    python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_* --ticker AAPL
    
    # Options analysis
    python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_* --ticker META --options

EXAMPLES:
    "Analyze META for a call option"
    "What do you think about TSLA right now?"
    "Should I buy NVDA?"
    "SPY entry analysis"

REQUIREMENTS:
    pip install yfinance ta pandas
============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
import ta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

console = Console()


# ============================================================================
# STOCK DATA FETCHING
# ============================================================================
def get_stock_data(ticker: str, period: str = "3mo") -> Tuple[pd.DataFrame, dict]:
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "META", "TSLA")
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        DataFrame with OHLCV data and info dict
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        info = stock.info
        return df, info
    except Exception as e:
        console.print(f"[red]Error fetching {ticker}: {e}[/red]")
        return pd.DataFrame(), {}


def get_extended_stock_data(ticker: str) -> dict:
    """
    Fetch extended stock data including fundamentals, analysts, and more.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with extended data
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Basic info
        data = {
            "company_name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "enterprise_value": info.get("enterpriseValue", 0),
        }
        
        # Price data
        data["current_price"] = info.get("currentPrice", info.get("regularMarketPrice", 0))
        data["previous_close"] = info.get("previousClose", 0)
        data["open"] = info.get("open", info.get("regularMarketOpen", 0))
        data["day_high"] = info.get("dayHigh", info.get("regularMarketDayHigh", 0))
        data["day_low"] = info.get("dayLow", info.get("regularMarketDayLow", 0))
        data["week_52_high"] = info.get("fiftyTwoWeekHigh", 0)
        data["week_52_low"] = info.get("fiftyTwoWeekLow", 0)
        data["fifty_day_avg"] = info.get("fiftyDayAverage", 0)
        data["two_hundred_day_avg"] = info.get("twoHundredDayAverage", 0)
        
        # Volume
        data["volume"] = info.get("volume", info.get("regularMarketVolume", 0))
        data["avg_volume"] = info.get("averageVolume", 0)
        data["avg_volume_10d"] = info.get("averageVolume10days", 0)
        
        # Fundamentals
        data["pe_ratio"] = info.get("trailingPE", 0)
        data["forward_pe"] = info.get("forwardPE", 0)
        data["peg_ratio"] = info.get("pegRatio", 0)
        data["price_to_book"] = info.get("priceToBook", 0)
        data["price_to_sales"] = info.get("priceToSalesTrailing12Months", 0)
        data["profit_margin"] = info.get("profitMargins", 0)
        data["operating_margin"] = info.get("operatingMargins", 0)
        data["revenue"] = info.get("totalRevenue", 0)
        data["revenue_growth"] = info.get("revenueGrowth", 0)
        data["earnings_growth"] = info.get("earningsGrowth", 0)
        data["free_cash_flow"] = info.get("freeCashflow", 0)
        
        # Dividends
        data["dividend_yield"] = info.get("dividendYield", 0)
        data["dividend_rate"] = info.get("dividendRate", 0)
        data["payout_ratio"] = info.get("payoutRatio", 0)
        data["ex_dividend_date"] = info.get("exDividendDate", None)
        
        # Analyst data
        data["target_high"] = info.get("targetHighPrice", 0)
        data["target_low"] = info.get("targetLowPrice", 0)
        data["target_mean"] = info.get("targetMeanPrice", 0)
        data["target_median"] = info.get("targetMedianPrice", 0)
        data["recommendation"] = info.get("recommendationKey", "N/A")
        data["recommendation_mean"] = info.get("recommendationMean", 0)
        data["num_analysts"] = info.get("numberOfAnalystOpinions", 0)
        
        # Short interest
        data["short_ratio"] = info.get("shortRatio", 0)
        data["short_percent"] = info.get("shortPercentOfFloat", 0)
        data["shares_short"] = info.get("sharesShort", 0)
        data["shares_outstanding"] = info.get("sharesOutstanding", 0)
        data["float_shares"] = info.get("floatShares", 0)
        
        # Earnings
        data["earnings_date"] = None
        try:
            calendar = stock.calendar
            if calendar is not None and not calendar.empty:
                if 'Earnings Date' in calendar.index:
                    earnings = calendar.loc['Earnings Date']
                    if hasattr(earnings, 'iloc'):
                        data["earnings_date"] = str(earnings.iloc[0])
                    else:
                        data["earnings_date"] = str(earnings)
        except:
            pass
        
        # Beta (volatility vs market)
        data["beta"] = info.get("beta", 0)
        
        # Institutional holdings
        data["held_by_institutions"] = info.get("heldPercentInstitutions", 0)
        data["held_by_insiders"] = info.get("heldPercentInsiders", 0)
        
        # Calculate percentage from 52-week levels
        if data["current_price"] and data["week_52_high"]:
            data["pct_from_52w_high"] = ((data["current_price"] - data["week_52_high"]) / data["week_52_high"]) * 100
        if data["current_price"] and data["week_52_low"]:
            data["pct_from_52w_low"] = ((data["current_price"] - data["week_52_low"]) / data["week_52_low"]) * 100
        
        # Get recent news
        try:
            news = stock.news[:5] if stock.news else []
            data["recent_news"] = [{"title": n.get("title", ""), "publisher": n.get("publisher", "")} for n in news]
        except:
            data["recent_news"] = []
        
        # Get insider transactions
        try:
            insider = stock.insider_transactions
            if insider is not None and not insider.empty:
                recent_insider = insider.head(5)
                data["insider_activity"] = recent_insider.to_dict('records') if not recent_insider.empty else []
            else:
                data["insider_activity"] = []
        except:
            data["insider_activity"] = []
        
        # Get institutional holders
        try:
            institutions = stock.institutional_holders
            if institutions is not None and not institutions.empty:
                data["top_institutions"] = institutions.head(5)['Holder'].tolist()
            else:
                data["top_institutions"] = []
        except:
            data["top_institutions"] = []
        
        return data
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch extended data: {e}[/yellow]")
        return {}


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicator columns
    """
    if df.empty or len(df) < 20:
        return df
    
    # RSI (14-period)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Moving Averages
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['EMA_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
    df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Lower'] = bollinger.bollinger_lband()
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # ATR (Average True Range) - for stop loss calculation
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    return df


def get_options_data(ticker: str) -> dict:
    """
    Fetch detailed options data for a stock.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with comprehensive options information
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get expiration dates
        expirations = stock.options[:5] if stock.options else []
        
        if not expirations:
            return {"available": False}
        
        # Get current price
        current_price = stock.info.get('currentPrice', stock.info.get('regularMarketPrice', 0))
        
        # Analyze multiple expirations
        all_options_data = []
        total_call_oi = 0
        total_put_oi = 0
        total_call_volume = 0
        total_put_volume = 0
        
        for exp in expirations[:3]:  # Analyze first 3 expirations
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
                
                # Sum up open interest and volume
                total_call_oi += calls['openInterest'].sum() if 'openInterest' in calls else 0
                total_put_oi += puts['openInterest'].sum() if 'openInterest' in puts else 0
                total_call_volume += calls['volume'].sum() if 'volume' in calls else 0
                total_put_volume += puts['volume'].sum() if 'volume' in puts else 0
                
                all_options_data.append({
                    'expiration': exp,
                    'calls': calls,
                    'puts': puts
                })
            except:
                continue
        
        # Get nearest expiration for detailed analysis
        nearest_exp = expirations[0]
        options_chain = stock.option_chain(nearest_exp)
        calls = options_chain.calls
        puts = options_chain.puts
        
        # Find ATM options
        atm_call = None
        atm_put = None
        if not calls.empty and current_price:
            calls['distance'] = abs(calls['strike'] - current_price)
            atm_call = calls.loc[calls['distance'].idxmin()]
            
            puts['distance'] = abs(puts['strike'] - current_price)
            atm_put = puts.loc[puts['distance'].idxmin()]
        
        # Find highest open interest strikes (key levels)
        max_call_oi_strike = calls.loc[calls['openInterest'].idxmax()]['strike'] if not calls.empty and 'openInterest' in calls else None
        max_put_oi_strike = puts.loc[puts['openInterest'].idxmax()]['strike'] if not puts.empty and 'openInterest' in puts else None
        
        # Find unusual activity (high volume relative to OI)
        unusual_calls = []
        unusual_puts = []
        
        if not calls.empty and 'volume' in calls and 'openInterest' in calls:
            calls['vol_oi_ratio'] = calls['volume'] / (calls['openInterest'] + 1)
            high_activity = calls[calls['vol_oi_ratio'] > 2].head(3)
            for _, row in high_activity.iterrows():
                unusual_calls.append({
                    'strike': row['strike'],
                    'volume': row['volume'],
                    'openInterest': row['openInterest'],
                    'iv': row.get('impliedVolatility', 0)
                })
        
        if not puts.empty and 'volume' in puts and 'openInterest' in puts:
            puts['vol_oi_ratio'] = puts['volume'] / (puts['openInterest'] + 1)
            high_activity = puts[puts['vol_oi_ratio'] > 2].head(3)
            for _, row in high_activity.iterrows():
                unusual_puts.append({
                    'strike': row['strike'],
                    'volume': row['volume'],
                    'openInterest': row['openInterest'],
                    'iv': row.get('impliedVolatility', 0)
                })
        
        # Calculate Put/Call ratio
        put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        put_call_volume_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        return {
            "available": True,
            "expirations": expirations,
            "nearest_expiration": nearest_exp,
            "atm_call": atm_call.to_dict() if atm_call is not None else None,
            "atm_put": atm_put.to_dict() if atm_put is not None else None,
            "current_price": current_price,
            "total_calls": len(calls),
            "total_puts": len(puts),
            "total_call_oi": int(total_call_oi),
            "total_put_oi": int(total_put_oi),
            "total_call_volume": int(total_call_volume),
            "total_put_volume": int(total_put_volume),
            "put_call_ratio": put_call_ratio,
            "put_call_volume_ratio": put_call_volume_ratio,
            "max_call_oi_strike": max_call_oi_strike,
            "max_put_oi_strike": max_put_oi_strike,
            "unusual_calls": unusual_calls,
            "unusual_puts": unusual_puts
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def get_historical_price_data(ticker: str) -> dict:
    """
    Fetch detailed historical price data for AI analysis.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with detailed price history
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get different timeframes
        df_1mo = stock.history(period="1mo", interval="1d")
        df_3mo = stock.history(period="3mo", interval="1d")
        df_1wk = stock.history(period="1mo", interval="1wk")
        
        history = {
            "daily_candles": [],
            "weekly_candles": [],
            "price_patterns": {},
            "key_dates": {}
        }
        
        # Last 30 daily candles with full OHLCV
        if not df_1mo.empty:
            for date, row in df_1mo.tail(30).iterrows():
                candle = {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(row['Open'], 2),
                    "high": round(row['High'], 2),
                    "low": round(row['Low'], 2),
                    "close": round(row['Close'], 2),
                    "volume": int(row['Volume']),
                    "change_pct": round(((row['Close'] - row['Open']) / row['Open']) * 100, 2)
                }
                # Determine candle type
                if row['Close'] > row['Open']:
                    candle['type'] = 'GREEN'
                elif row['Close'] < row['Open']:
                    candle['type'] = 'RED'
                else:
                    candle['type'] = 'DOJI'
                history["daily_candles"].append(candle)
        
        # Weekly candles
        if not df_1wk.empty:
            for date, row in df_1wk.iterrows():
                candle = {
                    "week_of": date.strftime("%Y-%m-%d"),
                    "open": round(row['Open'], 2),
                    "high": round(row['High'], 2),
                    "low": round(row['Low'], 2),
                    "close": round(row['Close'], 2),
                    "volume": int(row['Volume']),
                    "change_pct": round(((row['Close'] - row['Open']) / row['Open']) * 100, 2)
                }
                history["weekly_candles"].append(candle)
        
        # Analyze price patterns
        if len(df_1mo) >= 5:
            closes = df_1mo['Close'].values
            
            # Consecutive up/down days
            up_days = 0
            down_days = 0
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    up_days += 1
                    down_days = 0
                else:
                    down_days += 1
                    up_days = 0
            
            history["price_patterns"]["consecutive_up_days"] = up_days
            history["price_patterns"]["consecutive_down_days"] = down_days
            
            # Count total green vs red days this month
            green_days = sum(1 for i in range(1, len(closes)) if closes[i] > df_1mo['Open'].values[i])
            red_days = len(closes) - 1 - green_days
            history["price_patterns"]["green_days_this_month"] = green_days
            history["price_patterns"]["red_days_this_month"] = red_days
            
            # Identify gaps
            gaps = []
            for i in range(1, min(len(df_1mo), 30)):
                prev_close = df_1mo['Close'].iloc[i-1]
                curr_open = df_1mo['Open'].iloc[i]
                gap_pct = ((curr_open - prev_close) / prev_close) * 100
                if abs(gap_pct) > 1:  # Gap > 1%
                    gaps.append({
                        "date": df_1mo.index[i].strftime("%Y-%m-%d"),
                        "gap_pct": round(gap_pct, 2),
                        "type": "GAP UP" if gap_pct > 0 else "GAP DOWN"
                    })
            history["price_patterns"]["recent_gaps"] = gaps[-5:] if gaps else []
            
            # Highest and lowest days this month
            max_idx = df_1mo['High'].idxmax()
            min_idx = df_1mo['Low'].idxmin()
            history["key_dates"]["month_high_date"] = max_idx.strftime("%Y-%m-%d")
            history["key_dates"]["month_high_price"] = round(df_1mo['High'].max(), 2)
            history["key_dates"]["month_low_date"] = min_idx.strftime("%Y-%m-%d")
            history["key_dates"]["month_low_price"] = round(df_1mo['Low'].min(), 2)
            
            # Biggest single day moves
            df_1mo['daily_change'] = df_1mo['Close'].pct_change() * 100
            biggest_up = df_1mo['daily_change'].idxmax()
            biggest_down = df_1mo['daily_change'].idxmin()
            history["key_dates"]["biggest_up_day"] = {
                "date": biggest_up.strftime("%Y-%m-%d"),
                "change_pct": round(df_1mo['daily_change'].max(), 2)
            }
            history["key_dates"]["biggest_down_day"] = {
                "date": biggest_down.strftime("%Y-%m-%d"),
                "change_pct": round(df_1mo['daily_change'].min(), 2)
            }
            
            # Average daily range (volatility indicator)
            df_1mo['daily_range'] = ((df_1mo['High'] - df_1mo['Low']) / df_1mo['Low']) * 100
            history["price_patterns"]["avg_daily_range_pct"] = round(df_1mo['daily_range'].mean(), 2)
            
            # Opening vs closing tendency
            opens_higher_than_close = sum(1 for i in range(len(df_1mo)) if df_1mo['Open'].iloc[i] > df_1mo['Close'].iloc[i])
            history["price_patterns"]["opens_higher_than_close_days"] = opens_higher_than_close
            history["price_patterns"]["closes_higher_than_open_days"] = len(df_1mo) - opens_higher_than_close
        
        # Get intraday data - hourly for past 3 days
        try:
            df_hourly = stock.history(period="5d", interval="1h")
            if not df_hourly.empty:
                history["hourly_candles"] = []
                for date, row in df_hourly.tail(72).iterrows():  # Last ~3 days of hourly
                    candle = {
                        "datetime": date.strftime("%Y-%m-%d %H:%M"),
                        "open": round(row['Open'], 2),
                        "high": round(row['High'], 2),
                        "low": round(row['Low'], 2),
                        "close": round(row['Close'], 2),
                        "volume": int(row['Volume']),
                    }
                    if row['Close'] > row['Open']:
                        candle['type'] = 'GREEN'
                    elif row['Close'] < row['Open']:
                        candle['type'] = 'RED'
                    else:
                        candle['type'] = 'DOJI'
                    history["hourly_candles"].append(candle)
        except Exception as e:
            history["hourly_candles"] = []
        
        # Get intraday data - 15 minute for today
        try:
            df_15min = stock.history(period="1d", interval="15m")
            if not df_15min.empty:
                history["intraday_15min"] = []
                for date, row in df_15min.iterrows():
                    candle = {
                        "time": date.strftime("%H:%M"),
                        "open": round(row['Open'], 2),
                        "high": round(row['High'], 2),
                        "low": round(row['Low'], 2),
                        "close": round(row['Close'], 2),
                        "volume": int(row['Volume']),
                    }
                    if row['Close'] > row['Open']:
                        candle['type'] = 'GREEN'
                    elif row['Close'] < row['Open']:
                        candle['type'] = 'RED'
                    else:
                        candle['type'] = 'DOJI'
                    history["intraday_15min"].append(candle)
                
                # Calculate intraday stats
                if len(df_15min) > 0:
                    history["intraday_stats"] = {
                        "today_open": round(df_15min['Open'].iloc[0], 2),
                        "today_high": round(df_15min['High'].max(), 2),
                        "today_low": round(df_15min['Low'].min(), 2),
                        "current": round(df_15min['Close'].iloc[-1], 2),
                        "vwap": round((df_15min['Close'] * df_15min['Volume']).sum() / df_15min['Volume'].sum(), 2) if df_15min['Volume'].sum() > 0 else 0,
                        "total_volume": int(df_15min['Volume'].sum()),
                    }
        except Exception as e:
            history["intraday_15min"] = []
        
        # Get 3-month trend data
        if not df_3mo.empty:
            # Calculate monthly returns
            monthly_data = df_3mo.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            history["monthly_summary"] = []
            for date, row in monthly_data.iterrows():
                if pd.notna(row['Open']) and pd.notna(row['Close']):
                    history["monthly_summary"].append({
                        "month": date.strftime("%Y-%m"),
                        "open": round(row['Open'], 2),
                        "high": round(row['High'], 2),
                        "low": round(row['Low'], 2),
                        "close": round(row['Close'], 2),
                        "change_pct": round(((row['Close'] - row['Open']) / row['Open']) * 100, 2)
                    })
        
        return history
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch historical data: {e}[/yellow]")
        return {}


def calculate_support_resistance(df: pd.DataFrame, current_price: float) -> dict:
    """
    Calculate support and resistance levels to predict bottoms/tops.
    
    Args:
        df: DataFrame with OHLCV data
        current_price: Current stock price
        
    Returns:
        Dictionary with support/resistance levels
    """
    if df.empty or len(df) < 20:
        return {}
    
    # Get recent highs and lows
    recent_high = df['High'].tail(20).max()
    recent_low = df['Low'].tail(20).min()
    
    # Pivot points
    typical_price = (df['High'].iloc[-1] + df['Low'].iloc[-1] + df['Close'].iloc[-1]) / 3
    pivot = typical_price
    r1 = 2 * pivot - df['Low'].iloc[-1]
    s1 = 2 * pivot - df['High'].iloc[-1]
    r2 = pivot + (df['High'].iloc[-1] - df['Low'].iloc[-1])
    s2 = pivot - (df['High'].iloc[-1] - df['Low'].iloc[-1])
    
    # Fibonacci retracement levels
    fib_range = recent_high - recent_low
    fib_236 = recent_high - (fib_range * 0.236)
    fib_382 = recent_high - (fib_range * 0.382)
    fib_500 = recent_high - (fib_range * 0.500)
    fib_618 = recent_high - (fib_range * 0.618)
    fib_786 = recent_high - (fib_range * 0.786)
    
    # Find swing highs and lows (potential support/resistance)
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(df) - 2):
        # Swing high: higher than 2 candles before and after
        if (df['High'].iloc[i] > df['High'].iloc[i-1] and 
            df['High'].iloc[i] > df['High'].iloc[i-2] and
            df['High'].iloc[i] > df['High'].iloc[i+1] and 
            df['High'].iloc[i] > df['High'].iloc[i+2]):
            swing_highs.append(df['High'].iloc[i])
        
        # Swing low
        if (df['Low'].iloc[i] < df['Low'].iloc[i-1] and 
            df['Low'].iloc[i] < df['Low'].iloc[i-2] and
            df['Low'].iloc[i] < df['Low'].iloc[i+1] and 
            df['Low'].iloc[i] < df['Low'].iloc[i+2]):
            swing_lows.append(df['Low'].iloc[i])
    
    # Get nearest support and resistance
    supports = sorted([s for s in swing_lows + [s1, s2, fib_500, fib_618, fib_786] if s < current_price], reverse=True)[:3]
    resistances = sorted([r for r in swing_highs + [r1, r2, fib_236, fib_382] if r > current_price])[:3]
    
    # Predict potential bottom
    potential_bottom = min(supports) if supports else recent_low
    potential_top = max(resistances) if resistances else recent_high
    
    # Distance to support/resistance
    distance_to_support = ((current_price - potential_bottom) / current_price) * 100 if potential_bottom else 0
    distance_to_resistance = ((potential_top - current_price) / current_price) * 100 if potential_top else 0
    
    return {
        "pivot": pivot,
        "resistance_1": r1,
        "resistance_2": r2,
        "support_1": s1,
        "support_2": s2,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "fib_236": fib_236,
        "fib_382": fib_382,
        "fib_500": fib_500,
        "fib_618": fib_618,
        "supports": supports,
        "resistances": resistances,
        "potential_bottom": potential_bottom,
        "potential_top": potential_top,
        "distance_to_support_pct": distance_to_support,
        "distance_to_resistance_pct": distance_to_resistance
    }


# ============================================================================
# FORMAT DATA FOR AI
# ============================================================================
def format_stock_analysis(ticker: str, df: pd.DataFrame, info: dict, include_options: bool = False) -> str:
    """
    Format stock data as text for the AI model to analyze.
    
    Args:
        ticker: Stock symbol
        df: DataFrame with indicators
        info: Stock info dict
        include_options: Whether to include options data
        
    Returns:
        Formatted text analysis
    """
    if df.empty:
        return f"Could not fetch data for {ticker}"
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    # Get extended data
    ext_data = get_extended_stock_data(ticker)
    
    # Basic info
    company_name = ext_data.get('company_name', info.get('longName', info.get('shortName', ticker)))
    current_price = latest['Close']
    prev_close = prev['Close']
    change_pct = ((current_price - prev_close) / prev_close) * 100
    
    # Volume analysis
    volume = latest['Volume']
    avg_volume = info.get('averageVolume', df['Volume'].mean())
    volume_ratio = volume / avg_volume if avg_volume else 1
    
    # 52-week range
    week_52_high = info.get('fiftyTwoWeekHigh', df['High'].max())
    week_52_low = info.get('fiftyTwoWeekLow', df['Low'].min())
    price_vs_high = ((current_price - week_52_high) / week_52_high) * 100
    price_vs_low = ((current_price - week_52_low) / week_52_low) * 100
    
    # Format market cap
    market_cap = ext_data.get('market_cap', 0)
    if market_cap >= 1e12:
        market_cap_str = f"${market_cap/1e12:.2f}T"
    elif market_cap >= 1e9:
        market_cap_str = f"${market_cap/1e9:.2f}B"
    elif market_cap >= 1e6:
        market_cap_str = f"${market_cap/1e6:.2f}M"
    else:
        market_cap_str = f"${market_cap:,.0f}"
    
    # Build analysis text
    analysis = f"""
=== STOCK ANALYSIS: {ticker} ({company_name}) ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Sector: {ext_data.get('sector', 'N/A')} | Industry: {ext_data.get('industry', 'N/A')}

📊 PRICE ACTION:
- Current Price: ${current_price:.2f}
- Daily Change: {change_pct:+.2f}%
- Today's Range: ${latest['Low']:.2f} - ${latest['High']:.2f}
- 52-Week High: ${week_52_high:.2f} ({price_vs_high:+.1f}% from high)
- 52-Week Low: ${week_52_low:.2f} ({price_vs_low:+.1f}% from low)
- 50-Day Avg: ${ext_data.get('fifty_day_avg', 0):.2f} ({"ABOVE ✅" if current_price > ext_data.get('fifty_day_avg', 0) else "BELOW ❌"})
- 200-Day Avg: ${ext_data.get('two_hundred_day_avg', 0):.2f} ({"ABOVE ✅" if current_price > ext_data.get('two_hundred_day_avg', 0) else "BELOW ❌"})

📆 PERFORMANCE:
  - 1 Week: {((current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100) if len(df) >= 5 else 0:+.2f}%
  - 1 Month: {((current_price - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100) if len(df) >= 21 else 0:+.2f}%
  - 3 Month: {((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100):+.2f}%"""
    
    # Get detailed historical data
    hist_data = get_historical_price_data(ticker)
    
    if hist_data:
        # Add monthly summary
        if hist_data.get("monthly_summary"):
            analysis += "\n\n📅 MONTHLY BREAKDOWN (Last 3 Months):\n"
            for month in hist_data["monthly_summary"]:
                analysis += f"  {month['month']}: Open ${month['open']} → Close ${month['close']} ({month['change_pct']:+.1f}%) | High ${month['high']} | Low ${month['low']}\n"
        
        # Add price patterns
        if hist_data.get("price_patterns"):
            pp = hist_data["price_patterns"]
            analysis += f"""
📊 PRICE PATTERNS (This Month):
  - Green Days: {pp.get('green_days_this_month', 0)} | Red Days: {pp.get('red_days_this_month', 0)}
  - Consecutive Up Days: {pp.get('consecutive_up_days', 0)}
  - Consecutive Down Days: {pp.get('consecutive_down_days', 0)}
  - Avg Daily Range: {pp.get('avg_daily_range_pct', 0):.2f}%
"""
            # Recent gaps
            if pp.get('recent_gaps'):
                analysis += "  - Recent Gaps:\n"
                for gap in pp['recent_gaps']:
                    analysis += f"    • {gap['date']}: {gap['type']} {gap['gap_pct']:+.1f}%\n"
        
        # Key dates
        if hist_data.get("key_dates"):
            kd = hist_data["key_dates"]
            analysis += f"""
📌 KEY DATES THIS MONTH:
  - Month High: ${kd.get('month_high_price', 0)} on {kd.get('month_high_date', 'N/A')}
  - Month Low: ${kd.get('month_low_price', 0)} on {kd.get('month_low_date', 'N/A')}
  - Biggest Up Day: {kd.get('biggest_up_day', {}).get('change_pct', 0):+.1f}% on {kd.get('biggest_up_day', {}).get('date', 'N/A')}
  - Biggest Down Day: {kd.get('biggest_down_day', {}).get('change_pct', 0):+.1f}% on {kd.get('biggest_down_day', {}).get('date', 'N/A')}
"""
        
        # Daily candle data (last 10 days)
        if hist_data.get("daily_candles"):
            analysis += "\n📈 LAST 10 TRADING DAYS (Daily Candles):\n"
            analysis += "  Date       | Open     | High     | Low      | Close    | Change  | Type\n"
            analysis += "  " + "-" * 75 + "\n"
            for candle in hist_data["daily_candles"][-10:]:
                analysis += f"  {candle['date']} | ${candle['open']:<7} | ${candle['high']:<7} | ${candle['low']:<7} | ${candle['close']:<7} | {candle['change_pct']:+5.1f}% | {candle['type']}\n"
        
        # Weekly candle data
        if hist_data.get("weekly_candles"):
            analysis += "\n📊 WEEKLY CANDLES (Last 4 Weeks):\n"
            analysis += "  Week Of    | Open     | High     | Low      | Close    | Change\n"
            analysis += "  " + "-" * 65 + "\n"
            for candle in hist_data["weekly_candles"][-4:]:
                analysis += f"  {candle['week_of']} | ${candle['open']:<7} | ${candle['high']:<7} | ${candle['low']:<7} | ${candle['close']:<7} | {candle['change_pct']:+5.1f}%\n"
        
        # Intraday stats
        if hist_data.get("intraday_stats"):
            stats = hist_data["intraday_stats"]
            analysis += f"""
⏰ TODAY'S INTRADAY SUMMARY:
  - Open: ${stats.get('today_open', 0)}
  - High: ${stats.get('today_high', 0)}
  - Low: ${stats.get('today_low', 0)}
  - Current: ${stats.get('current', 0)}
  - VWAP: ${stats.get('vwap', 0)} ({"Price ABOVE VWAP ✅" if current_price > stats.get('vwap', 0) else "Price BELOW VWAP ❌"})
  - Total Volume: {stats.get('total_volume', 0):,}
"""
        
        # 15-minute candles for today
        if hist_data.get("intraday_15min"):
            analysis += "\n⏱️ TODAY'S 15-MINUTE CANDLES:\n"
            analysis += "  Time  | Open     | High     | Low      | Close    | Volume     | Type\n"
            analysis += "  " + "-" * 75 + "\n"
            for candle in hist_data["intraday_15min"]:
                analysis += f"  {candle['time']} | ${candle['open']:<7} | ${candle['high']:<7} | ${candle['low']:<7} | ${candle['close']:<7} | {candle['volume']:>10,} | {candle['type']}\n"
        
        # Hourly candles (last 24 hours)
        if hist_data.get("hourly_candles"):
            analysis += "\n🕐 HOURLY CANDLES (Last 24 Hours):\n"
            analysis += "  DateTime        | Open     | High     | Low      | Close    | Type\n"
            analysis += "  " + "-" * 70 + "\n"
            for candle in hist_data["hourly_candles"][-24:]:
                analysis += f"  {candle['datetime']} | ${candle['open']:<7} | ${candle['high']:<7} | ${candle['low']:<7} | ${candle['close']:<7} | {candle['type']}\n"
    
    # Calculate indicator signals
    rsi_val = latest['RSI']
    rsi_status = "OVERSOLD (potential buy signal)" if rsi_val < 30 else "OVERBOUGHT (potential sell signal)" if rsi_val > 70 else "NEUTRAL"
    macd_status = "BULLISH (MACD above signal)" if latest['MACD'] > latest['MACD_Signal'] else "BEARISH (MACD below signal)"
    sma20_status = "ABOVE ✅" if current_price > latest['SMA_20'] else "BELOW ❌"
    sma50_status = "ABOVE ✅" if current_price > latest['SMA_50'] else "BELOW ❌"
    ema_status = "BULLISH CROSS ✅" if latest['EMA_9'] > latest['EMA_21'] else "BEARISH CROSS ❌"
    bb_status = "NEAR UPPER (overbought zone)" if current_price > latest['BB_Upper'] * 0.98 else "NEAR LOWER (oversold zone)" if current_price < latest['BB_Lower'] * 1.02 else "MIDDLE RANGE"
    stoch_status = "OVERSOLD" if latest['Stoch_K'] < 20 else "OVERBOUGHT" if latest['Stoch_K'] > 80 else "NEUTRAL"
    vol_status = "HIGH VOLUME (confirms move)" if volume_ratio > 1.5 else "LOW VOLUME (weak conviction)" if volume_ratio < 0.5 else "NORMAL VOLUME"
    beta_val = ext_data.get('beta', 0)
    beta_status = "high volatility" if beta_val > 1.5 else "low volatility" if beta_val < 0.8 else "market-like"
    
    analysis += f"""

📈 TECHNICAL INDICATORS:

RSI (14): {rsi_val:.1f}
  → {rsi_status}

MACD:
  - MACD Line: {latest['MACD']:.3f}
  - Signal Line: {latest['MACD_Signal']:.3f}
  - Histogram: {latest['MACD_Histogram']:.3f}
  → {macd_status}

MOVING AVERAGES:
  - Price vs SMA 20: {sma20_status} (SMA 20: ${latest['SMA_20']:.2f})
  - Price vs SMA 50: {sma50_status} (SMA 50: ${latest['SMA_50']:.2f})
  - EMA 9 vs EMA 21: {ema_status}

BOLLINGER BANDS:
  - Upper: ${latest['BB_Upper']:.2f}
  - Middle: ${latest['BB_Middle']:.2f}
  - Lower: ${latest['BB_Lower']:.2f}
  - Price Position: {bb_status}

STOCHASTIC:
  - %K: {latest['Stoch_K']:.1f}
  - %D: {latest['Stoch_D']:.1f}
  → {stoch_status}

📊 VOLUME:
  - Today's Volume: {volume:,.0f}
  - Average Volume: {avg_volume:,.0f}
  - Volume Ratio: {volume_ratio:.2f}x average
  → {vol_status}

📉 RISK METRICS:
  - ATR (14): ${latest['ATR']:.2f}
  - Beta: {beta_val:.2f} ({beta_status})
  - Suggested Stop Loss: ${current_price - (latest['ATR'] * 2):.2f} (2x ATR below)
  - Suggested Take Profit: ${current_price + (latest['ATR'] * 3):.2f} (3x ATR above)

💰 FUNDAMENTALS:
  - Market Cap: {market_cap_str}
  - P/E Ratio: {ext_data.get('pe_ratio', 0):.2f} (Forward: {ext_data.get('forward_pe', 0):.2f})
  - PEG Ratio: {ext_data.get('peg_ratio', 0):.2f}
  - Price/Book: {ext_data.get('price_to_book', 0):.2f}
  - Profit Margin: {ext_data.get('profit_margin', 0)*100:.1f}%
  - Revenue Growth: {ext_data.get('revenue_growth', 0)*100:.1f}%
  - Earnings Growth: {ext_data.get('earnings_growth', 0)*100:.1f}%

📈 ANALYST RATINGS:
  - Recommendation: {ext_data.get('recommendation', 'N/A').upper()}
  - Price Target (Low): ${ext_data.get('target_low', 0):.2f}
  - Price Target (Mean): ${ext_data.get('target_mean', 0):.2f}
  - Price Target (High): ${ext_data.get('target_high', 0):.2f}
  - # of Analysts: {ext_data.get('num_analysts', 0)}
  - Upside to Mean Target: {((ext_data.get('target_mean', current_price) - current_price) / current_price * 100):.1f}%

🩳 SHORT INTEREST:
  - Short Ratio (Days to Cover): {ext_data.get('short_ratio', 0):.2f}
  - Short % of Float: {ext_data.get('short_percent', 0)*100:.2f}%
  - Shares Short: {ext_data.get('shares_short', 0):,}

🏛️ OWNERSHIP:
  - Institutional: {ext_data.get('held_by_institutions', 0)*100:.1f}%
  - Insiders: {ext_data.get('held_by_insiders', 0)*100:.1f}%
"""

    # Add earnings date if available
    if ext_data.get('earnings_date'):
        analysis += f"""
📅 UPCOMING EARNINGS: {ext_data.get('earnings_date')}
"""

    # Add dividend info if applicable
    if ext_data.get('dividend_yield', 0) > 0:
        analysis += f"""
💵 DIVIDEND:
  - Yield: {ext_data.get('dividend_yield', 0)*100:.2f}%
  - Annual Rate: ${ext_data.get('dividend_rate', 0):.2f}
  - Payout Ratio: {ext_data.get('payout_ratio', 0)*100:.1f}%
"""

    # Add recent news
    if ext_data.get('recent_news'):
        analysis += "\n📰 RECENT NEWS:\n"
        for news in ext_data.get('recent_news', [])[:3]:
            analysis += f"  • {news.get('title', 'N/A')[:80]}...\n"

    # Add insider activity summary
    if ext_data.get('insider_activity'):
        analysis += "\n👔 RECENT INSIDER ACTIVITY:\n"
        for insider in ext_data.get('insider_activity', [])[:3]:
            trans_type = insider.get('Text', 'Transaction')
            analysis += f"  • {trans_type}\n"

    # Add support/resistance analysis
    sr_levels = calculate_support_resistance(df, current_price)
    if sr_levels:
        analysis += f"""
🎯 SUPPORT & RESISTANCE (Bottom/Top Prediction):

KEY LEVELS:
  - Potential BOTTOM: ${sr_levels.get('potential_bottom', 0):.2f} ({sr_levels.get('distance_to_support_pct', 0):.1f}% below current)
  - Potential TOP: ${sr_levels.get('potential_top', 0):.2f} ({sr_levels.get('distance_to_resistance_pct', 0):.1f}% above current)

SUPPORT LEVELS (Buy Zones):
  - Support 1: ${sr_levels.get('support_1', 0):.2f}
  - Support 2: ${sr_levels.get('support_2', 0):.2f}
  - Recent Low: ${sr_levels.get('recent_low', 0):.2f}

RESISTANCE LEVELS (Sell Zones):
  - Resistance 1: ${sr_levels.get('resistance_1', 0):.2f}
  - Resistance 2: ${sr_levels.get('resistance_2', 0):.2f}
  - Recent High: ${sr_levels.get('recent_high', 0):.2f}

FIBONACCI LEVELS:
  - 23.6%: ${sr_levels.get('fib_236', 0):.2f}
  - 38.2%: ${sr_levels.get('fib_382', 0):.2f}
  - 50.0%: ${sr_levels.get('fib_500', 0):.2f}
  - 61.8%: ${sr_levels.get('fib_618', 0):.2f}
"""
    
    # Add options data if requested
    if include_options:
        options = get_options_data(ticker)
        if options.get("available"):
            analysis += f"""
📊 OPTIONS CHAIN ANALYSIS:

OVERVIEW:
  - Nearest Expiration: {options['nearest_expiration']}
  - Available Expirations: {', '.join(options['expirations'][:3])}
  - Total Call Open Interest: {options.get('total_call_oi', 0):,}
  - Total Put Open Interest: {options.get('total_put_oi', 0):,}
  - Put/Call OI Ratio: {options.get('put_call_ratio', 0):.2f} {"(BEARISH - more puts)" if options.get('put_call_ratio', 0) > 1.2 else "(BULLISH - more calls)" if options.get('put_call_ratio', 0) < 0.8 else "(NEUTRAL)"}

KEY STRIKE LEVELS:
  - Max Call OI Strike: ${options.get('max_call_oi_strike', 0):.2f} (potential resistance/magnet)
  - Max Put OI Strike: ${options.get('max_put_oi_strike', 0):.2f} (potential support/floor)
"""
            if options.get('atm_call'):
                call = options['atm_call']
                analysis += f"""
ATM CALL (Strike ${call.get('strike', 'N/A')}):
  - Premium: ${call.get('lastPrice', 0):.2f}
  - Bid/Ask: ${call.get('bid', 0):.2f} / ${call.get('ask', 0):.2f}
  - Implied Volatility: {call.get('impliedVolatility', 0)*100:.1f}%
  - Open Interest: {call.get('openInterest', 0):,}
  - Volume: {call.get('volume', 0):,}
"""
            if options.get('atm_put'):
                put = options['atm_put']
                analysis += f"""
ATM PUT (Strike ${put.get('strike', 'N/A')}):
  - Premium: ${put.get('lastPrice', 0):.2f}
  - Bid/Ask: ${put.get('bid', 0):.2f} / ${put.get('ask', 0):.2f}
  - Implied Volatility: {put.get('impliedVolatility', 0)*100:.1f}%
  - Open Interest: {put.get('openInterest', 0):,}
  - Volume: {put.get('volume', 0):,}
"""
            # Unusual options activity
            if options.get('unusual_calls'):
                analysis += "\n🔥 UNUSUAL CALL ACTIVITY (potential bullish bets):\n"
                for uc in options['unusual_calls']:
                    analysis += f"  - Strike ${uc['strike']}: Vol {uc['volume']:,} vs OI {uc['openInterest']:,}\n"
            
            if options.get('unusual_puts'):
                analysis += "\n🔥 UNUSUAL PUT ACTIVITY (potential bearish bets/hedging):\n"
                for up in options['unusual_puts']:
                    analysis += f"  - Strike ${up['strike']}: Vol {up['volume']:,} vs OI {up['openInterest']:,}\n"
    
    # Add summary
    bullish_signals = 0
    bearish_signals = 0
    
    if latest['RSI'] < 30: bullish_signals += 2
    elif latest['RSI'] > 70: bearish_signals += 2
    
    if latest['MACD'] > latest['MACD_Signal']: bullish_signals += 1
    else: bearish_signals += 1
    
    if current_price > latest['SMA_20']: bullish_signals += 1
    else: bearish_signals += 1
    
    if current_price > latest['SMA_50']: bullish_signals += 1
    else: bearish_signals += 1
    
    if volume_ratio > 1.2: 
        if change_pct > 0: bullish_signals += 1
        else: bearish_signals += 1
    
    if latest['Stoch_K'] < 20: bullish_signals += 1
    elif latest['Stoch_K'] > 80: bearish_signals += 1
    
    analysis += f"""
📋 SIGNAL SUMMARY:
  - Bullish Signals: {bullish_signals}
  - Bearish Signals: {bearish_signals}
  - Overall Bias: {"🟢 BULLISH" if bullish_signals > bearish_signals + 1 else "🔴 BEARISH" if bearish_signals > bullish_signals + 1 else "🟡 NEUTRAL"}
"""
    
    return analysis


# ============================================================================
# AI MODEL INTEGRATION
# ============================================================================
class TradingAssistant:
    """Trading assistant that combines real-time data with AI analysis."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the trading assistant.
        
        Args:
            model_path: Path to the fine-tuned model (optional)
        """
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load the fine-tuned AI model."""
        try:
            from inference import DocumentQA
            console.print(f"[dim]Loading AI model from {model_path}...[/dim]")
            self.model = DocumentQA(model_path=model_path)
            console.print("[green]✓ AI model loaded![/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not load AI model: {e}[/yellow]")
            console.print("[dim]Will provide analysis without AI recommendations.[/dim]")
            self.model = None
    
    def analyze(self, ticker: str, include_options: bool = False, question: str = None) -> str:
        """
        Analyze a stock and get AI recommendations.
        
        Args:
            ticker: Stock symbol
            include_options: Include options analysis
            question: Specific question to ask
            
        Returns:
            Analysis and recommendations
        """
        ticker = ticker.upper().strip()
        
        console.print(f"\n[bold blue]Fetching data for {ticker}...[/bold blue]")
        
        # Fetch and analyze data
        df, info = get_stock_data(ticker)
        if df.empty:
            return f"Could not fetch data for {ticker}. Please check the ticker symbol."
        
        df = calculate_indicators(df)
        analysis = format_stock_analysis(ticker, df, info, include_options)
        
        # Display the technical analysis
        console.print(Panel(analysis, title=f"[bold]{ticker} Technical Analysis[/bold]", border_style="blue"))
        
        # Get latest data for recommendations
        latest = df.iloc[-1]
        current_price = latest['Close']
        ext_data = get_extended_stock_data(ticker)
        sr_levels = calculate_support_resistance(df, current_price)
        
        # Calculate key signals
        rsi = latest['RSI']
        rsi_signal = 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
        macd_signal = 'BULLISH' if latest['MACD'] > latest['MACD_Signal'] else 'BEARISH'
        stoch = latest['Stoch_K']
        stoch_signal = 'OVERSOLD' if stoch < 20 else 'OVERBOUGHT' if stoch > 80 else 'NEUTRAL'
        sma20_signal = 'ABOVE' if current_price > latest['SMA_20'] else 'BELOW'
        sma50_signal = 'ABOVE' if current_price > latest['SMA_50'] else 'BELOW'
        bb_signal = 'NEAR LOWER (oversold)' if current_price < latest['BB_Lower'] * 1.02 else 'NEAR UPPER (overbought)' if current_price > latest['BB_Upper'] * 0.98 else 'MIDDLE'
        
        # Count bullish vs bearish
        bullish_count = sum([
            rsi < 30,  # Oversold
            latest['MACD'] > latest['MACD_Signal'],
            current_price > latest['SMA_20'],
            current_price > latest['SMA_50'],
            stoch < 20,
            current_price < latest['BB_Lower'] * 1.02
        ])
        bearish_count = sum([
            rsi > 70,
            latest['MACD'] < latest['MACD_Signal'],
            current_price < latest['SMA_20'],
            current_price < latest['SMA_50'],
            stoch > 80,
            current_price > latest['BB_Upper'] * 0.98
        ])
        
        overall_bias = "BULLISH" if bullish_count > bearish_count + 1 else "BEARISH" if bearish_count > bullish_count + 1 else "NEUTRAL"
        
        # Get AI recommendation if model is loaded
        if self.model:
            console.print("\n[dim]Getting AI analysis based on your trading documents...[/dim]")
            
            # Calculate price targets
            atr = latest['ATR']
            stop_loss = current_price - (atr * 2)
            take_profit_1 = current_price + (atr * 2)
            take_profit_2 = current_price + (atr * 3)
            analyst_target = ext_data.get('target_mean', current_price)
            upside = ((analyst_target - current_price) / current_price * 100) if current_price > 0 else 0
            
            # Calculate estimated timeframes based on ATR (typical daily move)
            # Assume price moves ~0.5-0.7 ATR per day on average in trending conditions
            avg_daily_move = atr * 0.6  # Conservative estimate
            
            dist_to_target1 = abs(take_profit_1 - current_price)
            dist_to_target2 = abs(take_profit_2 - current_price)
            dist_to_analyst = abs(analyst_target - current_price)
            
            days_to_t1 = max(1, int(dist_to_target1 / avg_daily_move)) if avg_daily_move > 0 else 5
            days_to_t2 = max(2, int(dist_to_target2 / avg_daily_move)) if avg_daily_move > 0 else 10
            days_to_analyst = max(5, int(dist_to_analyst / avg_daily_move)) if avg_daily_move > 0 else 30
            
            # Convert to readable timeframes
            def days_to_timeframe(days):
                if days <= 3:
                    return f"{days} days"
                elif days <= 7:
                    return "1 week"
                elif days <= 14:
                    return "1-2 weeks"
                elif days <= 21:
                    return "2-3 weeks"
                elif days <= 30:
                    return "1 month"
                elif days <= 60:
                    return "1-2 months"
                elif days <= 90:
                    return "2-3 months"
                else:
                    return f"{days // 30} months"
            
            tf_target1 = days_to_timeframe(days_to_t1)
            tf_target2 = days_to_timeframe(days_to_t2)
            tf_analyst = days_to_timeframe(days_to_analyst)
            
            # Build a directive prompt that forces specific output
            prompt = f"""### Instruction:
Analyze {ticker} stock data and provide a trading recommendation with specific price targets.

### Data:
{ticker} at ${current_price:.2f}
RSI: {rsi:.1f} ({rsi_signal}), MACD: {macd_signal}, Stochastic: {stoch:.1f} ({stoch_signal})
SMA20: ${latest['SMA_20']:.2f} ({sma20_signal}), SMA50: ${latest['SMA_50']:.2f} ({sma50_signal})
Bollinger: {bb_signal}, Bias: {overall_bias}
Support: ${sr_levels.get('support_1', 0):.2f}, Resistance: ${sr_levels.get('resistance_1', 0):.2f}
Analyst Target: ${analyst_target:.2f} ({upside:+.1f}% upside)

### Response:
**{ticker} RECOMMENDATION:** """

            try:
                ai_response = self.model.generate(
                    question=prompt,
                    max_new_tokens=300,  # Shorter to avoid rambling
                    temperature=0.4,
                    repetition_penalty=1.2,  # Prevent repetitive output
                    use_context=False,
                    raw_prompt=True  # Use prompt as-is, don't reformat
                )
                
                # Clean up - remove any prompt echo
                if "### Instruction:" in ai_response:
                    ai_response = ai_response.split("### Response:")[-1].strip()
                if ai_response.startswith(f"**{ticker} RECOMMENDATION:**"):
                    ai_response = ai_response[len(f"**{ticker} RECOMMENDATION:**"):].strip()
                
                # Cut off any document recitation or hallucinated content
                cutoff_phrases = ["[Page Break]", "Chapter ", "Section ", "Introduction to", "Conclusion:", "[Chart]", "[Image]", "[Graph]", "[Table]"]
                for phrase in cutoff_phrases:
                    if phrase in ai_response:
                        ai_response = ai_response.split(phrase)[0].strip()
                
                # Remove any remaining [Chart] or similar hallucinationssa
                import re
                ai_response = re.sub(r'\[Chart\]', '', ai_response)
                ai_response = re.sub(r'\[Image\]', '', ai_response)
                ai_response = re.sub(r'\[Graph\]', '', ai_response)
                ai_response = re.sub(r'\[Table\]', '', ai_response)
                ai_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', ai_response)  # Remove excessive newlines
                ai_response = ai_response.strip()
                
                # Handle empty or too short responses
                if not ai_response or len(ai_response.strip()) < 10:
                    return self._generate_rule_based_recommendation(ticker, df, current_price, sr_levels, ext_data)
                
                # Build formatted response with the AI analysis
                formatted_response = f"""## {ticker} Trading Analysis

**RECOMMENDATION:** {ai_response.split('.')[0].strip().upper() if ai_response else 'HOLD'}

{ai_response}

---
**Key Price Levels & Timeframes:**
- 📍 Current Price: ${current_price:.2f}
- 🎯 Entry Zone: ${current_price * 0.99:.2f} - ${current_price:.2f}
- 🛑 Stop Loss: ${stop_loss:.2f} ({((stop_loss - current_price) / current_price * 100):.1f}%)

**Targets:**
- ✅ Target 1: ${take_profit_1:.2f} ({((take_profit_1 - current_price) / current_price * 100):+.1f}%) → ⏱️ {tf_target1}
- ✅ Target 2: ${take_profit_2:.2f} ({((take_profit_2 - current_price) / current_price * 100):+.1f}%) → ⏱️ {tf_target2}
- 📊 Analyst Target: ${analyst_target:.2f} ({upside:+.1f}%) → ⏱️ {tf_analyst}

**Support/Resistance:**
- 📉 Support: ${sr_levels.get('support_1', 0):.2f}
- 📈 Resistance: ${sr_levels.get('resistance_1', 0):.2f}"""
                
                return formatted_response
            except Exception as e:
                console.print(f"[yellow]AI analysis error: {e}[/yellow]")
                return self._generate_rule_based_recommendation(ticker, df, current_price, sr_levels, ext_data)
        else:
            # Generate rule-based recommendation when no AI model is loaded
            return self._generate_rule_based_recommendation(ticker, df, current_price, sr_levels, ext_data)
    
    def _generate_rule_based_recommendation(self, ticker: str, df, current_price: float, sr_levels: dict, ext_data: dict) -> str:
        """Generate a recommendation based on technical indicators when AI fails or is unavailable."""
        latest = df.iloc[-1]
        
        # Count signals
        bullish = 0
        bearish = 0
        reasons_bull = []
        reasons_bear = []
        
        # RSI
        if latest['RSI'] < 30:
            bullish += 2
            reasons_bull.append(f"RSI is oversold at {latest['RSI']:.1f}")
        elif latest['RSI'] > 70:
            bearish += 2
            reasons_bear.append(f"RSI is overbought at {latest['RSI']:.1f}")
        
        # MACD
        if latest['MACD'] > latest['MACD_Signal']:
            bullish += 1
            reasons_bull.append("MACD is above signal line (bullish momentum)")
        else:
            bearish += 1
            reasons_bear.append("MACD is below signal line (bearish momentum)")
        
        # Moving averages
        if current_price > latest['SMA_20']:
            bullish += 1
            reasons_bull.append("Price above 20-day SMA")
        else:
            bearish += 1
            reasons_bear.append("Price below 20-day SMA")
        
        if current_price > latest['SMA_50']:
            bullish += 1
            reasons_bull.append("Price above 50-day SMA")
        else:
            bearish += 1
            reasons_bear.append("Price below 50-day SMA")
        
        # Stochastic
        if latest['Stoch_K'] < 20:
            bullish += 1
            reasons_bull.append(f"Stochastic oversold at {latest['Stoch_K']:.1f}")
        elif latest['Stoch_K'] > 80:
            bearish += 1
            reasons_bear.append(f"Stochastic overbought at {latest['Stoch_K']:.1f}")
        
        # Bollinger Bands
        if current_price < latest['BB_Lower'] * 1.02:
            bullish += 1
            reasons_bull.append("Price near lower Bollinger Band (potential bounce)")
        elif current_price > latest['BB_Upper'] * 0.98:
            bearish += 1
            reasons_bear.append("Price near upper Bollinger Band (potential pullback)")
        
        # Generate recommendation
        atr = latest['ATR']
        stop_loss = current_price - (atr * 2)
        take_profit_1 = current_price + (atr * 2)
        take_profit_2 = current_price + (atr * 3)
        analyst_target = ext_data.get('target_mean', current_price)
        
        # Calculate timeframes
        avg_daily_move = atr * 0.6
        days_to_t1 = max(1, int(abs(take_profit_1 - current_price) / avg_daily_move)) if avg_daily_move > 0 else 5
        days_to_t2 = max(2, int(abs(take_profit_2 - current_price) / avg_daily_move)) if avg_daily_move > 0 else 10
        days_to_analyst = max(5, int(abs(analyst_target - current_price) / avg_daily_move)) if avg_daily_move > 0 else 30
        
        def days_to_tf(days):
            if days <= 3: return f"{days} days"
            elif days <= 7: return "1 week"
            elif days <= 14: return "1-2 weeks"
            elif days <= 21: return "2-3 weeks"
            elif days <= 30: return "1 month"
            elif days <= 60: return "1-2 months"
            elif days <= 90: return "2-3 months"
            else: return f"{days // 30} months"
        
        if bullish > bearish + 1:
            signal = "🟢 **BUY (BULLISH)**"
            action = "Consider a LONG position (call option)"
        elif bearish > bullish + 1:
            signal = "🔴 **SELL/SHORT (BEARISH)**"
            action = "Consider a SHORT position (put option)"
            stop_loss = current_price + (atr * 2)
            take_profit_1 = current_price - (atr * 2)
            take_profit_2 = current_price - (atr * 3)
        else:
            signal = "🟡 **HOLD/WAIT (NEUTRAL)**"
            action = "Wait for clearer signals before entering"
        
        upside = ((analyst_target - current_price) / current_price * 100) if current_price > 0 else 0
        
        recommendation = f"""## {ticker} Trading Recommendation

### Signal: {signal}

### Key Levels & Timeframes:
- **Entry Price**: ${current_price:.2f}
- **Stop Loss**: ${stop_loss:.2f} ({abs((stop_loss - current_price) / current_price * 100):.1f}% risk)

### Targets:
- ✅ **Target 1**: ${take_profit_1:.2f} ({abs((take_profit_1 - current_price) / current_price * 100):.1f}%) → ⏱️ {days_to_tf(days_to_t1)}
- ✅ **Target 2**: ${take_profit_2:.2f} ({abs((take_profit_2 - current_price) / current_price * 100):.1f}%) → ⏱️ {days_to_tf(days_to_t2)}
- 📊 **Analyst Target**: ${analyst_target:.2f} ({upside:+.1f}%) → ⏱️ {days_to_tf(days_to_analyst)}
- **Risk/Reward Ratio**: 1:1.5

### Support/Resistance:
- 📉 Support: ${sr_levels.get('support_1', 0):.2f}
- 📈 Resistance: ${sr_levels.get('resistance_1', 0):.2f}

### Analyst Consensus: {ext_data.get('recommendation', 'N/A').upper()}

### Bullish Factors:
{chr(10).join(f'- {r}' for r in reasons_bull) if reasons_bull else '- None significant'}

### Bearish Factors:
{chr(10).join(f'- {r}' for r in reasons_bear) if reasons_bear else '- None significant'}

### Recommendation:
{action}

*Timeframes based on ATR (${atr:.2f}/day avg movement). Always do your own research.*"""
        
        return recommendation
    
    def chat(self):
        """Interactive chat mode."""
        console.print()
        console.print("[bold green]🤖 Trading Assistant[/bold green]")
        console.print("Ask me about any stock! Examples:")
        console.print("  • 'AAPL' or 'Analyze AAPL'")
        console.print("  • 'META call option' or 'META options'")
        console.print("  • 'Should I buy TSLA?'")
        console.print("  • 'NVDA bottom prediction'")
        console.print("  • 'SPY support levels'")
        console.print("Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye! Happy trading! 📈[/yellow]")
                    break
                
                # Parse the input
                ticker, include_options, question = self._parse_input(user_input)
                
                if ticker:
                    response = self.analyze(ticker, include_options, question)
                    console.print()
                    console.print(Panel(
                        Markdown(response),
                        title="[bold green]AI Recommendation[/bold green]",
                        border_style="green"
                    ))
                else:
                    console.print("[yellow]Please specify a stock ticker (e.g., AAPL, META, TSLA)[/yellow]")
                
                console.print()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Type 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def _parse_input(self, user_input: str) -> Tuple[str, bool, str]:
        """
        Parse user input to extract ticker and intent.
        
        Returns:
            (ticker, include_options, question)
        """
        text = user_input.upper()
        include_options = any(word in text for word in ['OPTION', 'CALL', 'PUT', 'OPTIONS'])
        
        # Common stock tickers
        common_tickers = [
            'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 
            'AMD', 'INTC', 'NFLX', 'DIS', 'BA', 'JPM', 'V', 'MA', 'WMT',
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'ARKK',
            'GME', 'AMC', 'PLTR', 'SOFI', 'NIO', 'RIVN', 'LCID',
            'COIN', 'HOOD', 'SQ', 'PYPL', 'SHOP', 'ROKU', 'SNAP',
            'UBER', 'LYFT', 'ABNB', 'DKNG', 'RBLX', 'U', 'CRWD',
            'ZM', 'DOCU', 'NET', 'SNOW', 'DDOG', 'MDB', 'OKTA'
        ]
        
        # Words to ignore (common English words that could look like tickers)
        ignore_words = {'I', 'A', 'AN', 'THE', 'TO', 'FOR', 'OF', 'IN', 'ON', 'AT', 
                       'BY', 'OR', 'AND', 'IS', 'IT', 'BE', 'DO', 'IF', 'SO', 'NO',
                       'YES', 'BUY', 'SELL', 'HOLD', 'GET', 'PUT', 'CALL', 'WHAT',
                       'SHOULD', 'ABOUT', 'THINK', 'STOCK', 'OPTION', 'OPTIONS'}
        
        # First, check for $ symbol (e.g., $TSLA)
        import re
        dollar_match = re.search(r'\$([A-Z]{1,5})', text)
        if dollar_match:
            return dollar_match.group(1), include_options, user_input
        
        # Try to find a known ticker first
        ticker = None
        words = text.replace(',', ' ').replace('.', ' ').replace('$', ' ').split()
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word in common_tickers:
                ticker = clean_word
                break
        
        # If no known ticker found, look for ticker-like words (but not common words)
        if not ticker:
            for word in words:
                clean_word = ''.join(c for c in word if c.isalpha())
                if (2 <= len(clean_word) <= 5 and 
                    clean_word.isalpha() and 
                    clean_word not in ignore_words):
                    ticker = clean_word
                    break
        
        return ticker, include_options, user_input


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Trading Assistant - AI-powered stock analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode
    python scripts/trading_assistant.py --model_path models/mistral-document-finetuned_*
    
    # Quick analysis
    python scripts/trading_assistant.py --ticker AAPL
    
    # With options
    python scripts/trading_assistant.py --ticker META --options
    
    # Without AI (just technical analysis)
    python scripts/trading_assistant.py --ticker TSLA --no_ai
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model for AI recommendations"
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Stock ticker to analyze (e.g., AAPL, META)"
    )
    
    parser.add_argument(
        "--options",
        action="store_true",
        help="Include options analysis"
    )
    
    parser.add_argument(
        "--no_ai",
        action="store_true",
        help="Skip AI analysis, just show technical data"
    )
    
    args = parser.parse_args()
    
    console.print("[bold magenta]" + "="*60 + "[/bold magenta]")
    console.print("[bold magenta]   🤖 AI Trading Assistant[/bold magenta]")
    console.print("[bold magenta]" + "="*60 + "[/bold magenta]")
    
    # Initialize assistant
    model_path = None if args.no_ai else args.model_path
    assistant = TradingAssistant(model_path=model_path)
    
    if args.ticker:
        # Single analysis mode
        response = assistant.analyze(args.ticker, args.options)
        console.print()
        console.print(Panel(
            Markdown(response),
            title="[bold green]AI Recommendation[/bold green]",
            border_style="green"
        ))
    else:
        # Interactive mode
        assistant.chat()


if __name__ == "__main__":
    main()

