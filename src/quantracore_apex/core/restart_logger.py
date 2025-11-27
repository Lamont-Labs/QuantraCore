"""
Restart logging module for QuantraCore Apex.

Logs system restarts and state transitions for audit purposes.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class RestartLogger:
    """
    Logs system restarts and state transitions.
    """
    
    def __init__(self, log_dir: str = "logs/scheduler"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new logging session.
        """
        self.start_time = datetime.utcnow()
        self.session_id = session_id or self.start_time.strftime("%Y%m%d_%H%M%S")
        
        self._log_event("session_start", {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
        })
        
        return self.session_id
    
    def log_restart(self, reason: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a system restart event.
        """
        self._log_event("restart", {
            "reason": reason,
            "metadata": metadata or {},
        })
    
    def log_state_transition(
        self,
        from_state: str,
        to_state: str,
        trigger: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a state transition event.
        """
        self._log_event("state_transition", {
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger,
            "metadata": metadata or {},
        })
    
    def end_session(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """
        End the current logging session.
        """
        self._log_event("session_end", {
            "session_id": self.session_id,
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            "summary": summary or {},
        })
    
    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Internal method to log an event.
        """
        log_entry = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "data": data,
        }
        
        filename = f"session_{self.session_id or 'unknown'}.jsonl"
        filepath = self.log_dir / filename
        
        with open(filepath, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")


restart_logger = RestartLogger()
