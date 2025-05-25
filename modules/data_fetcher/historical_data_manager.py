import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import requests
import ccxt
import asyncio
import aiohttp
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class HistoricalDataManager:
    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.sources = {
            'crypto': self._fetch_crypto_data,
            'forex': self._fetch_forex_data,
            'stocks': self._fetch_stock_data
        }
        
        # Initialize exchange connections
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'coinbase': ccxt.coinbasepro({'enableRateLimit': True}),
        }
        
    async def fetch_historical_data(self,
                                  symbol: str,
                                  timeframe: str,
                                  start_date: str,
                                  end_date: str = None,
                                  source: str = 'auto') -> pd.DataFrame:
        """Fetch historical data from specified source"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Determine appropriate source if auto
            if source == 'auto':
                source = self._determine_source(symbol)
            
            # Convert dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
            
            # Fetch data
            if source in self.sources:
                data = await self.sources[source](symbol, timeframe, start_dt, end_dt)
            else:
                raise ValueError(f"Unsupported data source: {source}")
            
            # Process and validate data
            if data is not None and not data.empty:
                data = self._process_data(data)
                self._save_to_cache(cache_key, data)
                self._save_to_file(symbol, timeframe, data)
                return data
            else:
                logger.warning(f"No data returned for {symbol} from {source}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
            
    async def _fetch_crypto_data(self, 
                               symbol: str,
                               timeframe: str,
                               start_dt: datetime,
                               end_dt: datetime) -> pd.DataFrame:
        """Fetch cryptocurrency historical data"""
        try:
            # Try multiple exchanges
            for exchange_name, exchange in self.exchanges.items():
                try:
                    # Convert timeframe to exchange format
                    tf_mapping = {
                        '1m': '1m', '5m': '5m', '15m': '15m',
                        '1h': '1h', '4h': '4h', 'D': '1d'
                    }
                    exchange_tf = tf_mapping.get(timeframe, '1h')
                    
                    # Fetch OHLCV data
                    ohlcv = await exchange.fetch_ohlcv(
                        symbol,
                        timeframe=exchange_tf,
                        since=int(start_dt.timestamp() * 1000),
                        limit=1000
                    )
                    
                    if ohlcv:
                        df = pd.DataFrame(ohlcv, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume'
                        ])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        return df
                        
                except Exception as e:
                    logger.warning(f"Error fetching from {exchange_name}: {e}")
                    continue
                    
            return None
            
        except Exception as e:
            logger.error(f"Error in crypto data fetch: {e}")
            return None

    async def _fetch_forex_data(self,
                              symbol: str,
                              timeframe: str,
                              start_dt: datetime,
                              end_dt: datetime) -> pd.DataFrame:
        """Fetch forex historical data"""
        try:
            # Use Alpha Vantage API for forex data
            api_key = self._get_api_key('alpha_vantage')
            if not api_key:
                logger.warning("No Alpha Vantage API key found")
                return None
                
            interval_mapping = {
                '1m': '1min', '5m': '5min', '15m': '15min',
                '1h': '60min', '4h': '240min', 'D': 'daily'
            }
            
            interval = interval_mapping.get(timeframe, 'daily')
            base, quote = symbol.split('/')
            
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "FX_INTRADAY" if interval != 'daily' else "FX_DAILY",
                "from_symbol": base,
                "to_symbol": quote,
                "interval": interval,
                "apikey": api_key,
                "outputsize": "full"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process the response
                        time_series_key = [k for k in data.keys() if 'Time Series' in k][0]
                        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
                        
                        # Rename columns
                        df.columns = ['open', 'high', 'low', 'close', 'volume']
                        df.index = pd.to_datetime(df.index)
                        
                        # Filter date range
                        df = df.loc[start_dt:end_dt]
                        return df
                        
            return None
            
        except Exception as e:
            logger.error(f"Error in forex data fetch: {e}")
            return None

    async def _fetch_stock_data(self,
                              symbol: str,
                              timeframe: str,
                              start_dt: datetime,
                              end_dt: datetime) -> pd.DataFrame:
        """Fetch stock market historical data"""
        try:
            # Use yfinance for stock data
            ticker = yf.Ticker(symbol)
            
            interval_mapping = {
                '1m': '1m', '5m': '5m', '15m': '15m',
                '1h': '1h', '4h': '4h', 'D': '1d'
            }
            interval = interval_mapping.get(timeframe, '1d')
            
            df = ticker.history(
                start=start_dt,
                end=end_dt,
                interval=interval
            )
            
            if not df.empty:
                # Standardize column names
                df.columns = [col.lower() for col in df.columns]
                return df
                
            return None
            
        except Exception as e:
            logger.error(f"Error in stock data fetch: {e}")
            return None

    def _determine_source(self, symbol: str) -> str:
        """Determine appropriate data source based on symbol"""
        if '/' in symbol:  # Crypto or Forex pair format
            if any(crypto in symbol for crypto in ['BTC', 'ETH', 'USDT']):
                return 'crypto'
            return 'forex'
        return 'stocks'  # Default to stocks for other formats

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate the data"""
        try:
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            # Handle missing values
            df.fillna(method='ffill', inplace=True)
            
            # Add calculated columns
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return df

    def _save_to_cache(self, key: str, data: pd.DataFrame):
        """Save data to cache"""
        self.cache[key] = data
        
        # Limit cache size
        if len(self.cache) > 100:
            oldest_key = min(self.cache.keys())
            del self.cache[oldest_key]

    def _save_to_file(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Save data to file"""
        try:
            filename = self.data_dir / f"{symbol}_{timeframe}.parquet"
            data.to_parquet(filename)
        except Exception as e:
            logger.error(f"Error saving data to file: {e}")

    def _get_api_key(self, service: str) -> Optional[str]:
        """Get API key from configuration"""
        try:
            config_file = Path("config/api_keys.json")
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                return config.get(service)
        except Exception as e:
            logger.error(f"Error reading API key: {e}")
        return None

    async def update_historical_data(self, symbols: List[str], timeframes: List[str]):
        """Update historical data for multiple symbols and timeframes"""
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                # Get last saved data
                filename = self.data_dir / f"{symbol}_{timeframe}.parquet"
                if filename.exists():
                    last_data = pd.read_parquet(filename)
                    start_date = last_data.index[-1]
                else:
                    start_date = datetime.now() - timedelta(days=365*2)  # 2 years
                
                # Create fetch task
                task = self.fetch_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    source='auto'
                )
                tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        success_count = sum(1 for r in results if isinstance(r, pd.DataFrame))
        logger.info(f"Updated {success_count}/{len(tasks)} datasets successfully") 