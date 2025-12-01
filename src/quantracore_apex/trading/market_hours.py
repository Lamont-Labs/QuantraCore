"""
Market Hours Service.

Provides real-time market session detection and extended hours support.

Extended Market Hours (Eastern Time):
- Pre-market: 4:00 AM - 9:30 AM ET
- Regular hours: 9:30 AM - 4:00 PM ET  
- After-hours: 4:00 PM - 8:00 PM ET
- Closed: 8:00 PM - 4:00 AM ET (and weekends/holidays)
"""

import logging
from datetime import datetime, time, date, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class MarketSession(str, Enum):
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


class MarketHoursService:
    """
    Market hours detection service for extended trading support.
    
    All times are in Eastern Time (ET/EDT):
    - Pre-market: 4:00 AM - 9:30 AM ET
    - Regular hours: 9:30 AM - 4:00 PM ET
    - After-hours: 4:00 PM - 8:00 PM ET
    - Closed: Outside above hours and weekends/holidays
    """
    
    PRE_MARKET_START = time(4, 0)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    AFTER_HOURS_END = time(20, 0)
    
    US_MARKET_HOLIDAYS_2025 = [
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
    ]
    
    EARLY_CLOSE_DATES_2025 = [
        date(2025, 7, 3),
        date(2025, 11, 28),
        date(2025, 12, 24),
    ]
    
    def __init__(self):
        self._cached_status: Optional[Dict[str, Any]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(seconds=5)
    
    def get_current_et_time(self) -> datetime:
        """Get current time in Eastern timezone."""
        return datetime.now(ET)
    
    def is_holiday(self, check_date: date) -> bool:
        """Check if date is a US market holiday."""
        return check_date in self.US_MARKET_HOLIDAYS_2025
    
    def is_weekend(self, check_date: date) -> bool:
        """Check if date is a weekend."""
        return check_date.weekday() >= 5
    
    def is_early_close(self, check_date: date) -> bool:
        """Check if market closes early (1 PM ET)."""
        return check_date in self.EARLY_CLOSE_DATES_2025
    
    def get_market_close_time(self, check_date: date) -> time:
        """Get market close time for a given date."""
        if self.is_early_close(check_date):
            return time(13, 0)
        return self.MARKET_CLOSE
    
    def get_after_hours_end_time(self, check_date: date) -> time:
        """Get after-hours end time for a given date."""
        if self.is_early_close(check_date):
            return time(17, 0)
        return self.AFTER_HOURS_END
    
    def get_current_session(self) -> MarketSession:
        """
        Determine the current market session.
        
        Returns:
            MarketSession: Current trading session
        """
        now = self.get_current_et_time()
        today = now.date()
        current_time = now.time()
        
        if self.is_weekend(today) or self.is_holiday(today):
            return MarketSession.CLOSED
        
        market_close = self.get_market_close_time(today)
        after_hours_end = self.get_after_hours_end_time(today)
        
        if current_time < self.PRE_MARKET_START:
            return MarketSession.CLOSED
        elif current_time < self.MARKET_OPEN:
            return MarketSession.PRE_MARKET
        elif current_time < market_close:
            return MarketSession.REGULAR
        elif current_time < after_hours_end:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED
    
    def is_trading_allowed(self, extended_hours: bool = True) -> bool:
        """
        Check if trading is allowed right now.
        
        Args:
            extended_hours: Whether to allow extended hours trading
            
        Returns:
            bool: True if trading is allowed
        """
        session = self.get_current_session()
        
        if session == MarketSession.CLOSED:
            return False
        
        if session == MarketSession.REGULAR:
            return True
        
        return extended_hours
    
    def get_time_until_next_session(self) -> Dict[str, Any]:
        """Get time until next market session."""
        now = self.get_current_et_time()
        today = now.date()
        current_time = now.time()
        session = self.get_current_session()
        
        if session == MarketSession.CLOSED:
            next_trading_day = self._get_next_trading_day(today, current_time)
            next_open = datetime.combine(next_trading_day, self.PRE_MARKET_START, tzinfo=ET)
            delta = next_open - now
            return {
                "next_session": "pre_market",
                "next_session_start": next_open.isoformat(),
                "hours_until": delta.total_seconds() / 3600,
            }
        elif session == MarketSession.PRE_MARKET:
            market_open = datetime.combine(today, self.MARKET_OPEN, tzinfo=ET)
            delta = market_open - now
            return {
                "next_session": "regular",
                "next_session_start": market_open.isoformat(),
                "hours_until": delta.total_seconds() / 3600,
            }
        elif session == MarketSession.REGULAR:
            market_close = datetime.combine(today, self.get_market_close_time(today), tzinfo=ET)
            delta = market_close - now
            return {
                "next_session": "after_hours",
                "next_session_start": market_close.isoformat(),
                "hours_until": delta.total_seconds() / 3600,
            }
        else:
            after_end = datetime.combine(today, self.get_after_hours_end_time(today), tzinfo=ET)
            delta = after_end - now
            next_day = self._get_next_trading_day(today + timedelta(days=1), time(0, 0))
            next_open = datetime.combine(next_day, self.PRE_MARKET_START, tzinfo=ET)
            return {
                "next_session": "closed",
                "next_session_start": after_end.isoformat(),
                "hours_until_close": delta.total_seconds() / 3600,
                "next_trading_day": next_open.isoformat(),
            }
    
    def _get_next_trading_day(self, from_date: date, from_time: time) -> date:
        """Get the next trading day from a given date."""
        check_date = from_date
        
        if from_time >= self.AFTER_HOURS_END:
            check_date += timedelta(days=1)
        
        while self.is_weekend(check_date) or self.is_holiday(check_date):
            check_date += timedelta(days=1)
        
        return check_date
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive market hours status.
        
        Returns cached result if available and fresh.
        """
        now = datetime.utcnow()
        
        if (self._cached_status and self._cache_time and 
            now - self._cache_time < self._cache_duration):
            return self._cached_status
        
        et_now = self.get_current_et_time()
        session = self.get_current_session()
        
        session_display = {
            MarketSession.PRE_MARKET: "Pre-Market",
            MarketSession.REGULAR: "Regular Hours",
            MarketSession.AFTER_HOURS: "After-Hours",
            MarketSession.CLOSED: "Closed",
        }
        
        session_info = self.get_time_until_next_session()
        
        status = {
            "current_session": session.value,
            "session_display": session_display[session],
            "is_extended_hours": session in [MarketSession.PRE_MARKET, MarketSession.AFTER_HOURS],
            "trading_allowed": self.is_trading_allowed(extended_hours=True),
            "trading_allowed_regular_only": self.is_trading_allowed(extended_hours=False),
            "current_time_et": et_now.strftime("%Y-%m-%d %H:%M:%S ET"),
            "current_time_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "is_weekend": self.is_weekend(et_now.date()),
            "is_holiday": self.is_holiday(et_now.date()),
            "is_early_close": self.is_early_close(et_now.date()),
            "next_session_info": session_info,
            "session_schedule": {
                "pre_market": f"{self.PRE_MARKET_START.strftime('%H:%M')} - {self.MARKET_OPEN.strftime('%H:%M')} ET",
                "regular": f"{self.MARKET_OPEN.strftime('%H:%M')} - {self.get_market_close_time(et_now.date()).strftime('%H:%M')} ET",
                "after_hours": f"{self.get_market_close_time(et_now.date()).strftime('%H:%M')} - {self.get_after_hours_end_time(et_now.date()).strftime('%H:%M')} ET",
            },
            "extended_hours_enabled": True,
            "note": "All times in Eastern Time (ET). Extended hours trading is fully supported.",
        }
        
        self._cached_status = status
        self._cache_time = now
        
        return status


_market_hours_service: Optional[MarketHoursService] = None


def get_market_hours_service() -> MarketHoursService:
    """Get singleton market hours service."""
    global _market_hours_service
    if _market_hours_service is None:
        _market_hours_service = MarketHoursService()
    return _market_hours_service
