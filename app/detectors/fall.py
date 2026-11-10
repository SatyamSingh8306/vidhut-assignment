from collections.abc import Sequence


class FallClassifier:
    """Confirms a horizontal body pose across consecutive frames."""

    def __init__(self, confirm_frames: int) -> None:
        self.confirm_frames = confirm_frames
        self.horizontal_frames = 0

    def update(self, box: Sequence[float] | None) -> bool:
        if box is None:
            self.horizontal_frames = 0
            return False
        x1, y1, x2, y2 = box
        horizontal = x2 - x1 > (y2 - y1) * 1.15
        self.horizontal_frames = self.horizontal_frames + 1 if horizontal else 0
        if self.horizontal_frames < self.confirm_frames:
            return False
        self.horizontal_frames = 0
        return True

