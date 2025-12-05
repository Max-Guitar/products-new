import time
import math
import streamlit as st

class ETAEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.placeholder = st.empty()
        self.last_fraction: float | None = None
        self.last_update_time: float | None = None

    def estimate_text(self) -> str:
        now = time.time()
        elapsed = now - self.start_time
        if self.last_fraction and self.last_fraction > 0:
            base_remaining = (elapsed / self.last_fraction) - elapsed
            buffer = max(20, elapsed * 0.1)  # буфер: 10% от прошедшего времени, минимум 20 сек
            remaining = base_remaining + buffer + 60.0
            if remaining < 60:
                return f"ETA: ~{int(remaining)} sec"
            else:
                minutes = math.ceil(remaining / 60.0)
                return f"ETA: ~{minutes} minute{'s' if minutes != 1 else ''}"
        return "ETA: calculating..."

    def update(self, fraction_done: float) -> None:
        self.last_fraction = fraction_done
        now = time.time()
        if self.last_update_time is None or (now - self.last_update_time) >= 60:
            self.placeholder.markdown(f"**{self.estimate_text()}**")
            self.last_update_time = now

    def estimate_minutes(self) -> int:
        now = time.time()
        elapsed = now - self.start_time
        if self.last_fraction and self.last_fraction > 0:
            remaining = (elapsed / self.last_fraction) - elapsed
            return math.ceil(remaining / 60.0)
        return -1

    def close(self) -> None:
        self.placeholder.empty()
