import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import ccxt
from typing import Dict, List, Optional
import traceback
import psutil
import sys
import gc

class AutoDebugger:
    def __init__(self):
        self.logger = self._setup_logger()
        self.test_data_sources = {
            'crypto': ['BTC/USDT', 'ETH/USDT'],
            'forex': ['EUR/USD', 'GBP/USD'],
            'stocks': ['AAPL', 'GOOGL'],
            'commodities': ['GC=F', 'SI=F']  # Gold and Silver futures
        }
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', 'D']
        self.test_results = {}
        
    def _setup_logger(self):
        logger = logging.getLogger('AutoDebugger')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler('debug_logs/system_debug.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    async def run_full_system_test(self):
        """Run comprehensive system tests"""
        try:
            self.logger.info("Starting full system diagnostic...")
            
            # Test components in sequence
            await self._test_data_fetching()
            await self._test_ai_core()
            await self._test_trading_logic()
            await self._test_risk_management()
            await self._test_performance()
            
            # Generate report
            self._generate_debug_report()
            
        except Exception as e:
            self.logger.error(f"Error in full system test: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def _test_data_fetching(self):
        """Test data fetching from free sources"""
        self.logger.info("Testing data fetching capabilities...")
        
        results = {
            'success_rate': 0,
            'failed_sources': [],
            'latency': {}
        }
        
        try:
            # Test YFinance
            for symbol in ['AAPL', 'GOOGL']:
                start_time = datetime.now()
                data = yf.download(symbol, period='1d', interval='1m')
                latency = (datetime.now() - start_time).total_seconds()
                
                if not data.empty:
                    results['success_rate'] += 1
                    results['latency'][symbol] = latency
                else:
                    results['failed_sources'].append(f'YFinance-{symbol}')

            # Test Crypto data (using CCXT with public API)
            exchange = ccxt.binance({'enableRateLimit': True})
            for symbol in ['BTC/USDT', 'ETH/USDT']:
                try:
                    start_time = datetime.now()
                    ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=100)
                    latency = (datetime.now() - start_time).total_seconds()
                    
                    if ohlcv:
                        results['success_rate'] += 1
                        results['latency'][symbol] = latency
                except Exception as e:
                    results['failed_sources'].append(f'CCXT-{symbol}')
                    self.logger.warning(f"Error fetching {symbol}: {str(e)}")

            self.test_results['data_fetching'] = results
            self.logger.info(f"Data fetching test results: {results}")
            
        except Exception as e:
            self.logger.error(f"Error in data fetching test: {str(e)}")
            self.test_results['data_fetching'] = {'error': str(e)}

    async def _test_ai_core(self):
        """Test AI core functionality"""
        self.logger.info("Testing AI core components...")
        
        try:
            # Test model loading and basic predictions
            from modules.synapse_ai_core import SynapseAI  # Import your AI core
            
            test_data = self._generate_test_data()
            ai_results = {
                'model_loading': False,
                'prediction_capability': False,
                'memory_usage': 0,
                'processing_time': 0
            }
            
            # Test model initialization
            start_time = datetime.now()
            try:
                ai_core = SynapseAI()
                ai_results['model_loading'] = True
            except Exception as e:
                self.logger.error(f"Error loading AI model: {str(e)}")
            
            # Test prediction capabilities
            if ai_results['model_loading']:
                try:
                    prediction = ai_core.predict(test_data)
                    ai_results['prediction_capability'] = True
                    ai_results['processing_time'] = (datetime.now() - start_time).total_seconds()
                except Exception as e:
                    self.logger.error(f"Error in AI prediction: {str(e)}")
            
            # Monitor memory usage
            process = psutil.Process()
            ai_results['memory_usage'] = process.memory_info().rss / 1024 / 1024  # in MB
            
            self.test_results['ai_core'] = ai_results
            self.logger.info(f"AI core test results: {ai_results}")
            
        except Exception as e:
            self.logger.error(f"Error in AI core test: {str(e)}")
            self.test_results['ai_core'] = {'error': str(e)}

    async def _test_trading_logic(self):
        """Test trading logic without real trades"""
        self.logger.info("Testing trading logic...")
        
        try:
            test_data = self._generate_test_data()
            trading_results = {
                'signal_generation': False,
                'position_sizing': False,
                'order_creation': False,
                'risk_checks': False
            }
            
            # Test signal generation
            try:
                signals = self._test_signal_generation(test_data)
                trading_results['signal_generation'] = True
            except Exception as e:
                self.logger.error(f"Error in signal generation: {str(e)}")
            
            # Test position sizing
            if trading_results['signal_generation']:
                try:
                    position_sizes = self._test_position_sizing(signals)
                    trading_results['position_sizing'] = True
                except Exception as e:
                    self.logger.error(f"Error in position sizing: {str(e)}")
            
            self.test_results['trading_logic'] = trading_results
            self.logger.info(f"Trading logic test results: {trading_results}")
            
        except Exception as e:
            self.logger.error(f"Error in trading logic test: {str(e)}")
            self.test_results['trading_logic'] = {'error': str(e)}

    async def _test_risk_management(self):
        """Test risk management systems"""
        self.logger.info("Testing risk management systems...")
        
        try:
            risk_results = {
                'position_limits': False,
                'exposure_checks': False,
                'stop_loss': False,
                'volatility_adjustment': False
            }
            
            test_portfolio = {
                'BTC/USDT': {'position': 1.0, 'entry_price': 40000},
                'ETH/USDT': {'position': 10.0, 'entry_price': 2000}
            }
            
            # Test position limits
            try:
                max_position = self._check_position_limits(test_portfolio)
                risk_results['position_limits'] = True
            except Exception as e:
                self.logger.error(f"Error in position limits check: {str(e)}")
            
            # Test exposure checks
            try:
                exposure = self._check_total_exposure(test_portfolio)
                risk_results['exposure_checks'] = True
            except Exception as e:
                self.logger.error(f"Error in exposure check: {str(e)}")
            
            self.test_results['risk_management'] = risk_results
            self.logger.info(f"Risk management test results: {risk_results}")
            
        except Exception as e:
            self.logger.error(f"Error in risk management test: {str(e)}")
            self.test_results['risk_management'] = {'error': str(e)}

    async def _test_performance(self):
        """Test system performance and resource usage"""
        self.logger.info("Testing system performance...")
        
        try:
            performance_results = {
                'cpu_usage': 0,
                'memory_usage': 0,
                'response_time': 0,
                'gc_stats': {}
            }
            
            # Monitor CPU usage
            performance_results['cpu_usage'] = psutil.cpu_percent(interval=1)
            
            # Monitor memory usage
            process = psutil.Process()
            performance_results['memory_usage'] = process.memory_info().rss / 1024 / 1024  # in MB
            
            # Test response time
            start_time = datetime.now()
            await self._test_system_response()
            performance_results['response_time'] = (datetime.now() - start_time).total_seconds()
            
            # Garbage collection stats
            gc.collect()
            performance_results['gc_stats'] = {
                'garbage': len(gc.garbage),
                'collections': gc.get_count()
            }
            
            self.test_results['performance'] = performance_results
            self.logger.info(f"Performance test results: {performance_results}")
            
        except Exception as e:
            self.logger.error(f"Error in performance test: {str(e)}")
            self.test_results['performance'] = {'error': str(e)}

    def _generate_debug_report(self):
        """Generate comprehensive debug report"""
        self.logger.info("Generating debug report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / (1024 * 1024 * 1024)  # in GB
            },
            'test_results': self.test_results,
            'recommendations': []
        }
        
        # Analyze results and generate recommendations
        if self.test_results.get('data_fetching', {}).get('failed_sources'):
            report['recommendations'].append(
                "Some data sources failed - consider implementing fallback sources"
            )
        
        if self.test_results.get('performance', {}).get('memory_usage', 0) > 1000:  # if > 1GB
            report['recommendations'].append(
                "High memory usage detected - consider optimizing memory management"
            )
        
        # Save report
        with open('debug_logs/debug_report.json', 'w') as f:
            import json
            json.dump(report, f, indent=4)
        
        self.logger.info("Debug report generated successfully")
        return report

    def _generate_test_data(self) -> pd.DataFrame:
        """Generate synthetic test data"""
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
        data = pd.DataFrame(index=dates)
        data['open'] = np.random.normal(100, 2, len(dates))
        data['high'] = data['open'] + abs(np.random.normal(0, 0.5, len(dates)))
        data['low'] = data['open'] - abs(np.random.normal(0, 0.5, len(dates)))
        data['close'] = np.random.normal(100, 2, len(dates))
        data['volume'] = np.random.normal(1000000, 200000, len(dates))
        return data

    def _test_signal_generation(self, data: pd.DataFrame) -> Dict:
        """Test trading signal generation"""
        signals = {
            'buy': [],
            'sell': [],
            'hold': []
        }
        
        # Simple moving average crossover for testing
        data['SMA20'] = data['close'].rolling(window=20).mean()
        data['SMA50'] = data['close'].rolling(window=50).mean()
        
        for i in range(len(data)):
            if i < 50:
                continue
            if data['SMA20'].iloc[i] > data['SMA50'].iloc[i] and \
               data['SMA20'].iloc[i-1] <= data['SMA50'].iloc[i-1]:
                signals['buy'].append(i)
            elif data['SMA20'].iloc[i] < data['SMA50'].iloc[i] and \
                 data['SMA20'].iloc[i-1] >= data['SMA50'].iloc[i-1]:
                signals['sell'].append(i)
            else:
                signals['hold'].append(i)
                
        return signals

    def _test_position_sizing(self, signals: Dict) -> Dict:
        """Test position sizing logic"""
        test_balance = 100000  # Test with 100k USD
        risk_per_trade = 0.02  # 2% risk per trade
        
        position_sizes = {
            'buy': [],
            'sell': []
        }
        
        for buy_signal in signals['buy']:
            position_size = test_balance * risk_per_trade
            position_sizes['buy'].append(position_size)
            
        for sell_signal in signals['sell']:
            position_size = test_balance * risk_per_trade
            position_sizes['sell'].append(position_size)
            
        return position_sizes

    def _check_position_limits(self, portfolio: Dict) -> bool:
        """Check if positions are within limits"""
        max_position_size = 100000  # Example: 100k USD max position
        
        for symbol, position in portfolio.items():
            if abs(position['position'] * position['entry_price']) > max_position_size:
                return False
        return True

    def _check_total_exposure(self, portfolio: Dict) -> float:
        """Calculate total portfolio exposure"""
        total_exposure = sum(
            abs(pos['position'] * pos['entry_price']) 
            for pos in portfolio.values()
        )
        return total_exposure

    async def _test_system_response(self):
        """Test system response time with dummy operations"""
        await asyncio.sleep(0.1)  # Simulate some async operations
        _ = self._generate_test_data()  # Some CPU work
        _ = gc.collect()  # Some memory management

if __name__ == "__main__":
    debugger = AutoDebugger()
    asyncio.run(debugger.run_full_system_test()) 