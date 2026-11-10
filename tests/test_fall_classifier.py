from app.detectors.fall import FallClassifier


def test_requires_persistent_horizontal_pose() -> None:
    classifier = FallClassifier(confirm_frames=3)
    horizontal = (0, 0, 200, 100)
    assert classifier.update(horizontal) is False
    assert classifier.update(horizontal) is False
    assert classifier.update(horizontal) is True


def test_upright_pose_resets_confirmation() -> None:
    classifier = FallClassifier(confirm_frames=2)
    assert classifier.update((0, 0, 200, 100)) is False
    assert classifier.update((0, 0, 100, 200)) is False
    assert classifier.update((0, 0, 200, 100)) is False

