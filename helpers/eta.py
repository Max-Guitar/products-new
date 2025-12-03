import time
import math
import streamlit as st

class ETAEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.last_update = self.start_time
        self.placeholder = st.empty()
        self.last_fraction: float | None = None

    def update(self, fraction_done: float) -> None:
        self.last_fraction = fraction_done
        now = time.time()
        if now - self.last_update >= 60:
            if 0 < fraction_done < 1:
                elapsed = now - self.start_time
                remaining = (elapsed / fraction_done) - elapsed
                minutes = math.ceil(remaining / 60.0)
                self.placeholder.markdown(f"**ETA:** ~{minutes} minute{'s' if minutes != 1 else ''}")
            self.last_update = now

    def estimate_minutes(self) -> int:
        now = time.time()
        elapsed = now - self.start_time
        if self.last_fraction and self.last_fraction > 0:
            remaining = (elapsed / self.last_fraction) - elapsed
            return math.ceil(remaining / 60.0)
        return -1

    def close(self) -> None:
        self.placeholder.empty()
