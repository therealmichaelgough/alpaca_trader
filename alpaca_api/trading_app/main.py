import time
import json
import logging
import threading
import argparse
import os
import sys
from collections import deque
import pandas as pd
import boto3
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class TradingEngine:
    def __init__(self, config):
        """
        Initialize the trading engine with the provided configuration.

        Args:
            config (dict): Configuration dictionary loaded from the config file
        """
        self.config = config

        # Initialize logging
        self._setup_logging()

        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=self.config['alpaca']['api_key'],
            secret_key=self.config['alpaca']['secret_key'],
            paper=self.config['alpaca'].get('paper', True)  # Default to paper trading for safety
        )
        self.data_client = StockHistoricalDataClient(
            api_key=self.config['alpaca']['api_key'],
            secret_key=self.config['alpaca']['secret_key']
        )

        # Initialize S3 client
        if 's3' in self.config:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=self.config['aws'].get('access_key'),
                aws_secret_access_key=self.config['aws'].get('secret_key'),
                region_name=self.config['aws'].get('region', 'us-east-1')
            )
        else:
            self.s3 = None
            logging.warning("S3 configuration not provided. Strategy updates will be disabled.")

        # Initialize market data cache
        self.data_cache = {}  # symbol -> deque of candlesticks
        self.buffer_size = self.config.get('buffer_size', 30)

        # Initialize strategy
        self.strategy = None
        self.strategy_last_updated = 0

        # Initialize threads
        self.market_data_thread = threading.Thread(target=self._market_data_worker)
        self.strategy_update_thread = threading.Thread(target=self._strategy_update_worker)
        self.signal_processing_thread = threading.Thread(target=self._signal_processing_worker)

        self.running = False

    def _setup_logging(self):
        """Configure logging based on the config settings"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file')

        logging_handlers = []
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_handler = logging.FileHandler(log_file)
            logging_handlers.append(file_handler)

        # Always add console handler
        console_handler = logging.StreamHandler()
        logging_handlers.append(console_handler)

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=logging_handlers
        )

    def start(self):
        """Start the trading engine and all worker threads"""
        self.running = True
        self._load_latest_strategy()
        self.market_data_thread.start()
        self.strategy_update_thread.start()
        self.signal_processing_thread.start()
        logging.info("Trading engine started")

    def stop(self):
        """Stop the trading engine and all worker threads"""
        self.running = False
        logging.info("Stopping trading engine...")

        # Wait for threads to finish with timeout
        self.market_data_thread.join(timeout=5)
        self.strategy_update_thread.join(timeout=5)
        self.signal_processing_thread.join(timeout=5)

        logging.info("Trading engine stopped")

    def _load_latest_strategy(self):
        """Load the latest strategy from local file or S3 bucket"""
        try:
            # First try loading from S3 if configured
            if self.s3 and 's3' in self.config:
                logging.info(f"Loading strategy from S3 bucket: {self.config['s3']['bucket']}")
                response = self.s3.get_object(
                    Bucket=self.config['s3']['bucket'],
                    Key=self.config['s3']['strategy_key']
                )
                strategy_data = json.loads(response['Body'].read().decode('utf-8'))
            # Fall back to local file if specified
            elif 'strategy_file' in self.config:
                logging.info(f"Loading strategy from local file: {self.config['strategy_file']}")
                with open(self.config['strategy_file'], 'r') as f:
                    strategy_data = json.load(f)
            else:
                logging.error("No strategy source configured")
                return

            # Validate strategy data
            if not self._validate_strategy(strategy_data):
                logging.error("Invalid strategy data")
                return

            self.strategy = strategy_data
            self.strategy_last_updated = time.time()

            # Initialize data cache for new symbols
            for symbol in self.strategy['symbols']:
                if symbol not in self.data_cache:
                    self.data_cache[symbol] = deque(maxlen=self.buffer_size)
                    self._initialize_symbol_data(symbol)

            logging.info(f"Loaded strategy: {self.strategy['id']} v{self.strategy['version']}")
        except Exception as e:
            logging.error(f"Error loading strategy: {e}")

    def _validate_strategy(self, strategy):
        """Validate the structure of a strategy definition"""
        required_fields = ['id', 'version', 'symbols', 'signals']
        if not all(field in strategy for field in required_fields):
            logging.error(f"Strategy missing required fields: {required_fields}")
            return False

        if not isinstance(strategy['symbols'], list) or len(strategy['symbols']) == 0:
            logging.error("Strategy must include at least one symbol")
            return False

        if not isinstance(strategy['signals'], list) or len(strategy['signals']) == 0:
            logging.error("Strategy must include at least one signal definition")
            return False

        return True

    def _initialize_symbol_data(self, symbol):
        """Load initial historical data for a symbol"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=self.buffer_size
            )
            bars = self.data_client.get_stock_bars(request)

            if not bars or symbol not in bars:
                logging.warning(f"No historical data available for {symbol}")
                return

            for bar in bars[symbol]:
                self.data_cache[symbol].append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })

            logging.info(f"Initialized historical data for {symbol} with {len(self.data_cache[symbol])} candles")
        except Exception as e:
            logging.error(f"Error initializing data for {symbol}: {e}")

    def _market_data_worker(self):
        """Worker thread to fetch market data"""
        logging.info("Market data worker started")
        while self.running:
            try:
                # Skip if market is closed
                clock = self.trading_client.get_clock()
                if not clock.is_open and not self.config.get('run_when_market_closed', False):
                    logging.info("Market is closed, sleeping for 60 seconds")
                    time.sleep(60)
                    continue

                # Fetch latest minute candles for all symbols
                if self.strategy:
                    for symbol in self.strategy['symbols']:
                        request = StockBarsRequest(
                            symbol_or_symbols=symbol,
                            timeframe=TimeFrame.Minute,
                            limit=1
                        )
                        bars = self.data_client.get_stock_bars(request)

                        if not bars or symbol not in bars or len(bars[symbol]) == 0:
                            logging.warning(f"No recent data available for {symbol}")
                            continue

                        for bar in bars[symbol]:
                            candle = {
                                'timestamp': bar.timestamp,
                                'open': bar.open,
                                'high': bar.high,
                                'low': bar.low,
                                'close': bar.close,
                                'volume': bar.volume
                            }

                            # Add to cache
                            if symbol not in self.data_cache:
                                self.data_cache[symbol] = deque(maxlen=self.buffer_size)
                            self.data_cache[symbol].append(candle)
                            logging.debug(f"Updated data for {symbol}: {candle}")

                # Sleep until next minute
                time.sleep(self.config.get('market_data_interval', 60))
            except Exception as e:
                logging.error(f"Error in market data worker: {e}")
                time.sleep(10)  # Back off on error

        logging.info("Market data worker stopped")

    def _strategy_update_worker(self):
        """Worker thread to check for strategy updates"""
        logging.info("Strategy update worker started")
        while self.running:
            try:
                if self.s3 and 's3' in self.config:
                    # Check for updates at the specified interval
                    self._load_latest_strategy()
                    time.sleep(self.config.get('strategy_update_interval', 300))
                else:
                    # If no S3 configuration, this thread has nothing to do
                    logging.info("No S3 configuration for strategy updates, worker going to sleep")
                    # Sleep for a long time, effectively disabling the thread
                    time.sleep(3600)
            except Exception as e:
                logging.error(f"Error in strategy update worker: {e}")
                time.sleep(60)  # Back off on error

        logging.info("Strategy update worker stopped")

    def _signal_processing_worker(self):
        """Worker thread to process signals and execute trades"""
        logging.info("Signal processing worker started")
        while self.running:
            try:
                # Skip if market is closed
                clock = self.trading_client.get_clock()
                if not clock.is_open and not self.config.get('trade_when_market_closed', False):
                    time.sleep(60)
                    continue

                if not self.strategy:
                    logging.warning("No strategy loaded, skipping signal processing")
                    time.sleep(10)
                    continue

                # Process signals for each symbol
                for symbol in self.strategy['symbols']:
                    if symbol in self.data_cache and len(self.data_cache[symbol]) > 0:
                        self._process_symbol(symbol)

                # Sleep for a few seconds
                time.sleep(self.config.get('signal_processing_interval', 5))
            except Exception as e:
                logging.error(f"Error in signal processing worker: {e}")
                time.sleep(10)  # Back off on error

        logging.info("Signal processing worker stopped")

    def _process_symbol(self, symbol):
        """Process signals for a given symbol"""
        try:
            # Convert deque to DataFrame for easier processing
            candles = list(self.data_cache[symbol])
            df = pd.DataFrame(candles)

            # Skip if we don't have enough data
            if len(df) < self.config.get('min_candles_required', 5):
                logging.debug(f"Not enough data for {symbol}, skipping signal processing")
                return

            # Get current position for this symbol
            try:
                position = self.trading_client.get_open_position(symbol)
                has_position = True
                position_qty = position.qty
                position_side = "LONG" if float(position_qty) > 0 else "SHORT"
            except Exception:
                # No position exists
                has_position = False
                position_qty = 0
                position_side = None

            # Calculate signals based on strategy
            signals = self._calculate_signals(df, symbol)

            # Execute trades based on signals
            for signal in signals:
                self._execute_trade(symbol, signal, has_position, position_qty, position_side)

        except Exception as e:
            logging.error(f"Error processing symbol {symbol}: {e}")

    def _calculate_signals(self, df, symbol):
        """
        Calculate trading signals based on strategy

        Args:
            df (pandas.DataFrame): DataFrame containing candle data
            symbol (str): The symbol to calculate signals for

        Returns:
            list: List of signal dictionaries
        """
        # This is a placeholder - you would implement your actual signal logic here
        # based on the strategy definition

        # For demonstration purposes, returning an empty list (no signals)
        return []

    def _execute_trade(self, symbol, signal, has_position, position_qty, position_side):
        """
        Execute a trade based on the signal

        Args:
            symbol (str): The symbol to trade
            signal (dict): The signal dictionary with action and sizing
            has_position (bool): Whether we already have a position
            position_qty (float): Current position quantity
            position_side (str): Current position side ("LONG", "SHORT", or None)
        """
        # This is a placeholder - you would implement the actual order execution
        # using the Alpaca Trading API
        pass


def load_config(config_path):
    """
    Load the configuration file

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def parse_arguments():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Alpaca Trading System')

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Override logging level from config file'
    )

    parser.add_argument(
        '--paper',
        action='store_true',
        help='Override to use paper trading regardless of config setting'
    )

    parser.add_argument(
        '--strategy-file',
        type=str,
        help='Override strategy file path from config'
    )

    return parser.parse_args()


def main():
    """Main entry point for the trading system"""
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Override configuration with command line arguments if provided
    if args.log_level:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['level'] = args.log_level

    if args.paper:
        if 'alpaca' not in config:
            config['alpaca'] = {}
        config['alpaca']['paper'] = True

    if args.strategy_file:
        config['strategy_file'] = args.strategy_file

    # Create and start the trading engine
    engine = TradingEngine(config)
    engine.start()

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        engine.stop()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()