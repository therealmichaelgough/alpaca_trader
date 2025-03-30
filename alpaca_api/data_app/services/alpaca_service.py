import logging
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)


class AlpacaService:
    """Service class for interacting with Alpaca APIs."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca clients.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Whether to use paper trading (default: True)
        """
        if not api_key or not secret_key:
            logger.warning("Alpaca API credentials not provided")

        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper
        )

        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )

        # Map of string timeframe to Alpaca TimeFrame enum
        self.timeframe_map = {
            "1min": TimeFrame.Minute,
            "5min": TimeFrame.Minute,
            "15min": TimeFrame.Minute,
            "1hour": TimeFrame.Hour,
            "1day": TimeFrame.Day,
        }

        # Multipliers for timeframes that need them
        self.multiplier_map = {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "1hour": 1,
            "1day": 1,
        }

    def get_stock_bars(
            self,
            symbol: str,
            timeframe: str,
            start: datetime,
            end: datetime,
            limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get historical bar/candle data for a symbol.

        Args:
            symbol: Stock symbol (e.g., AAPL)
            timeframe: Time interval (1Min, 5Min, 15Min, 1Hour, 1Day)
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars to return

        Returns:
            List of bar data dictionaries
        """
        # Normalize timeframe string
        timeframe_lower = timeframe.lower()

        # Get the appropriate Alpaca TimeFrame enum
        if timeframe_lower not in self.timeframe_map:
            logger.warning(f"Unsupported timeframe: {timeframe}, defaulting to 1Day")
            alpaca_timeframe = TimeFrame.Day
            multiplier = 1
        else:
            alpaca_timeframe = self.timeframe_map[timeframe_lower]
            multiplier = self.multiplier_map[timeframe_lower]

        # Create request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=alpaca_timeframe,
            start=start,
            end=end,
            limit=limit,
            adjustment='all'
        )

        # Add multiplier for timeframes that need it
        if multiplier > 1:
            request_params.timeframe = alpaca_timeframe(multiplier)

        # Request data from Alpaca
        try:
            bars_response = self.data_client.get_stock_bars(request_params)

            # Check if data was returned
            if not bars_response or symbol not in bars_response:
                logger.warning(f"No data returned for {symbol}")
                return []

            # Convert Alpaca response to list of dictionaries
            bars_list = []
            for bar in bars_response[symbol]:
                bars_list.append({
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume
                })

            return bars_list

        except Exception as e:
            logger.error(f"Error fetching bar data for {symbol}: {str(e)}")
            raise

    def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get latest quote for a symbol.

        Args:
            symbol: Stock symbol (e.g., AAPL)

        Returns:
            Dictionary with quote data
        """
        request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)

        try:
            quote_response = self.data_client.get_stock_latest_quote(request_params)

            if not quote_response or symbol not in quote_response:
                logger.warning(f"No quote data returned for {symbol}")
                return {}

            quote = quote_response[symbol]
            return {
                "ask_price": quote.ask_price,
                "ask_size": quote.ask_size,
                "bid_price": quote.bid_price,
                "bid_size": quote.bid_size,
                "timestamp": quote.timestamp.isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            raise

    def get_pe_ratio(self, symbol: str, days: int = 365) -> Dict[str, Any]:
        """
        Get P/E ratio data for a symbol.

        Note: This is a simplified implementation that estimates P/E ratios.
        In a production environment, you would use a financial data API that
        provides actual earnings data.

        Args:
            symbol: Stock symbol (e.g., AAPL)
            days: Number of days of historical data

        Returns:
            Dictionary with P/E ratio data and current P/E
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get historical price data (monthly)
            bars = self.get_stock_bars(
                symbol=symbol,
                timeframe="1Day",
                start=start_date,
                end=end_date
            )

            if not bars:
                return {"data": [], "current_pe": 0}

            # Generate synthetic earnings data (in a real app, you would fetch this)
            # This is just for demonstration

            # Different companies have different typical P/E ratios
            # Use the symbol to seed a "typical" P/E ratio
            ticker_seed = sum(ord(c) for c in symbol) % 100
            base_pe_ratio = 10 + (ticker_seed % 30)  # Range from 10 to 39

            pe_data = []

            # Process daily bars into monthly data points for P/E
            monthly_data = {}
            for bar in bars:
                date = bar["timestamp"].split("T")[0]
                year_month = date[:7]  # YYYY-MM

                if year_month not in monthly_data:
                    monthly_data[year_month] = {"close": bar["close"], "date": date}
                else:
                    # Just keep the last day of each month
                    if date > monthly_data[year_month]["date"]:
                        monthly_data[year_month] = {"close": bar["close"], "date": date}

            # Calculate P/E ratios for each month
            for year_month, data in monthly_data.items():
                # Add some variance to make it realistic
                noise = 0.9 + (np.random.random() * 0.2)  # 0.9 to 1.1
                pe_ratio = base_pe_ratio * noise

                pe_data.append({
                    "date": data["date"],
                    "pe": round(pe_ratio, 2)
                })

            # Sort by date
            pe_data.sort(key=lambda x: x["date"])

            return {
                "data": pe_data,
                "current_pe": pe_data[-1]["pe"] if pe_data else 0
            }

        except Exception as e:
            logger.error(f"Error calculating P/E ratio for {symbol}: {str(e)}")
            raise