from numpy import float32
from numpy.typing import NDArray


class TrackableObject:
    """
    An object that is being tracked.
    """

    def __init__(self, object_id: int, centroid: NDArray[float32]):
        self.object_id = object_id
        self.centroid = centroid

        self.counted = False
