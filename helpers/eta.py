import time
import math
import streamlit as st

class ETAEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.placeholder = st.empty()
        self.last_fraction: float | None = None

    def estimate_text(self) -> str:
        now = time.time()
        elapsed = now - self.start_time
        if self.last_fraction and self.last_fraction > 0:
            remaining = (elapsed / self.last_fraction) - elapsed
            if remaining < 60:
                return f"ETA: ~{int(remaining)} sec"
            else:
                minutes = math.ceil(remaining / 60.0)
                return f"ETA: ~{minutes} minute{'s' if minutes != 1 else ''}"
        return "ETA: calculating..."

    def update(self, fraction_done: float) -> None:
        self.last_fraction = fraction_done
        self.placeholder.markdown(f"**{self.estimate_text()}**")

    def estimate_minutes(self) -> int:
        now = time.time()
        elapsed = now - self.start_time
        if self.last_fraction and self.last_fraction > 0:
            remaining = (elapsed / self.last_fraction) - elapsed
            return math.ceil(remaining / 60.0)
        return -1

    def close(self) -> None:
        self.placeholder.empty()
