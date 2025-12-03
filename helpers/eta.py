import time
import math
import streamlit as st

class ETAEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.last_update = self.start_time
        self.placeholder = st.empty()

    def update(self, fraction_done: float) -> None:
        now = time.time()
        if now - self.last_update >= 60:
            if 0 < fraction_done < 1:
                elapsed = now - self.start_time
                remaining = (elapsed / fraction_done) - elapsed
                minutes = math.ceil(remaining / 60.0)
                self.placeholder.markdown(f"**ETA:** ~{minutes} minute{'s' if minutes != 1 else ''}")
            self.last_update = now

    def close(self) -> None:
        self.placeholder.empty()
