from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from app.config import get_settings
from app.services.alpaca_service import AlpacaService
from app.services.indicator_service import IndicatorService
from app.models.schemas import (
    StockBarsResponse,
    StockQuoteResponse,
    PERatioResponse,
    TechnicalIndicatorResponse,
    TechnicalIndicatorRequest
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Analysis API",
    description="API for stock market data and technical analysis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you should specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get Alpaca service
def get_alpaca_service():
    settings = get_settings()
    return AlpacaService(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        paper=settings.alpaca_paper
    )


# Dependency to get Indicator service
def get_indicator_service():
    return IndicatorService()


@app.get("/")
async def root():
    return {"message": "Stock Analysis API is running"}


@app.get("/api/stocks/{ticker}/bars", response_model=StockBarsResponse)
async def get_stock_bars(
        ticker: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100,
        alpaca_service: AlpacaService = Depends(get_alpaca_service)
):
    """
    Get stock price bars/candles for a specified ticker.

    - **ticker**: Stock symbol (e.g., AAPL, MSFT)
    - **timeframe**: Time interval (1Min, 5Min, 15Min, 1Hour, 1Day)
    - **start**: Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
    - **end**: End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
    - **limit**: Maximum number of bars to return
    """
    try:
        # Calculate default date range if not provided
        if not end:
            end_date = datetime.now()
        else:
            end_date = datetime.fromisoformat(end.replace('Z', '+00:00'))

        if not start:
            if timeframe.lower() in ['1min', '5min', '15min']:
                start_date = end_date - timedelta(days=1)
            elif timeframe.lower() == '1hour':
                start_date = end_date - timedelta(days=7)
            else:
                start_date = end_date - timedelta(days=365)
        else:
            start_date = datetime.fromisoformat(start.replace('Z', '+00:00'))

        # Get stock bars from Alpaca service
        bars = alpaca_service.get_stock_bars(
            symbol=ticker.upper(),
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            limit=limit
        )

        return StockBarsResponse(ticker=ticker.upper(), bars=bars)

    except Exception as e:
        logger.error(f"Error getting stock bars for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{ticker}/quote", response_model=StockQuoteResponse)
async def get_stock_quote(
        ticker: str,
        alpaca_service: AlpacaService = Depends(get_alpaca_service)
):
    """
    Get the latest quote for a specified ticker.

    - **ticker**: Stock symbol (e.g., AAPL, MSFT)
    """
    try:
        quote = alpaca_service.get_stock_quote(symbol=ticker.upper())
        return StockQuoteResponse(ticker=ticker.upper(), quote=quote)

    except Exception as e:
        logger.error(f"Error getting stock quote for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{ticker}/pe-ratio", response_model=PERatioResponse)
async def get_pe_ratio(
        ticker: str,
        days: int = 365,
        alpaca_service: AlpacaService = Depends(get_alpaca_service)
):
    """
    Get historical P/E ratio data for a specified ticker.

    - **ticker**: Stock symbol (e.g., AAPL, MSFT)
    - **days**: Number of days of historical data to fetch
    """
    try:
        pe_data = alpaca_service.get_pe_ratio(
            symbol=ticker.upper(),
            days=days
        )

        return PERatioResponse(
            ticker=ticker.upper(),
            data=pe_data.get("data", []),
            current_pe=pe_data.get("current_pe", 0)
        )

    except Exception as e:
        logger.error(f"Error getting P/E ratio for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/indicators/calculate", response_model=TechnicalIndicatorResponse)
async def calculate_indicator(
        request: TechnicalIndicatorRequest,
        alpaca_service: AlpacaService = Depends(get_alpaca_service),
        indicator_service: IndicatorService = Depends(get_indicator_service)
):
    """
    Calculate technical indicators for a specified ticker.

    - **ticker**: Stock symbol (e.g., AAPL, MSFT)
    - **indicator**: Indicator type (RSI, MACD, SMA, EMA, Bollinger)
    - **params**: Parameters for the indicator calculation
    - **timeframe**: Time interval (1Min, 5Min, 15Min, 1Hour, 1Day)
    - **start_date**: Start date (YYYY-MM-DD)
    - **end_date**: End date (YYYY-MM-DD)
    """
    try:
        # First get price data from Alpaca
        bars = alpaca_service.get_stock_bars(
            symbol=request.ticker,
            timeframe=request.timeframe,
            start=datetime.fromisoformat(request.start_date),
            end=datetime.fromisoformat(request.end_date) if request.end_date else datetime.now(),
            limit=request.limit or 1000
        )

        # Calculate the requested indicator
        result = indicator_service.calculate_indicator(
            bars=bars,
            indicator_type=request.indicator,
            params=request.params
        )

        return TechnicalIndicatorResponse(
            ticker=request.ticker,
            indicator=request.indicator,
            data=result
        )

    except Exception as e:
        logger.error(f"Error calculating {request.indicator} for {request.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__serve__":
    import uvicorn

    uvicorn.run("app.serve:app", host="0.0.0.0", port=8000, reload=True)