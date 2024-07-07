class TrackableObject:
    """
    An object that is being tracked.
    """

    def __init__(self, object_id: int, centroid: tuple[int, int]):
        self.object_id = object_id
        self.centroid = centroid

        self.counted = False
