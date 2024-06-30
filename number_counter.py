import threading
from logging import getLogger


class NumberCounter:
    """
    A simple class that keeps track of a number and allows incrementing and decrementing it in a thread-safe manner.
    """

    def __init__(self):
        self.count = 0

        self._lock = threading.Lock()
        self._logger = getLogger(__name__)

    def increment(self, amount: int = 1):
        with self._lock:
            self.count += amount

        self._logger.info(f"Incremented count by {amount} to {self.count}")

    def decrement(self, amount: int = 1):
        with self._lock:
            self.count -= amount

        self._logger.info(f"Decremented count by {amount} to {self.count}")
