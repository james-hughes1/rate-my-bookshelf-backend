from io import BytesIO

import numpy as np
from PIL import Image

from app.services.visualize import (
    create_mean_value_image,
    create_segmentation_gif,
    draw_segment_boundaries,
    visualize_selected_segment,
)


# -----------------------------
# Helpers
# -----------------------------
def dummy_image(w=10, h=10):
    """Create a small dummy RGB image."""
    return np.ones((h, w, 3), dtype=np.uint8) * 100


def dummy_mask(w=10, h=10, x1=2, y1=2, x2=7, y2=7):
    """Create a square mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


# -----------------------------
# draw_segment_boundaries
# -----------------------------
def test_draw_segment_boundaries():
    image = dummy_image()
    masks = [dummy_mask()]

    result = draw_segment_boundaries(image, masks)

    # Should not be identical because boundaries were drawn
    assert not np.array_equal(result, image)
    assert result.shape == image.shape


# -----------------------------
# create_mean_value_image
# -----------------------------
def test_create_mean_value_image():
    image = dummy_image()
    mask = dummy_mask()
    masks = [mask]

    result = create_mean_value_image(image, masks)

    # Inside mask should equal original mean
    mean_val = image[mask > 0].mean(axis=0)

    assert np.all(result[mask > 0] == mean_val.astype(np.uint8))
    # Outside mask should remain zero
    assert np.all(result[mask == 0] == 0)


# -----------------------------
# visualize_selected_segment
# -----------------------------
def test_visualize_selected_segment_basic():
    image = dummy_image()
    masks = [dummy_mask()]
    selected = 0

    result = visualize_selected_segment(image, masks, selected)

    # Should remain an RGB image
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_visualize_selected_segment_invalid_index():
    image = dummy_image()
    masks = [dummy_mask()]

    # Index out of range should just return mean-value image
    result = visualize_selected_segment(image, masks, selected_mask_idx=5)

    expected = create_mean_value_image(image, masks)
    assert np.array_equal(result, expected)


# -----------------------------
# create_segmentation_gif
# -----------------------------
class DummySegResult:
    """Minimal fake segmentation result object used only for testing."""

    def __init__(self):
        self.masks = [dummy_mask()]
        # depth_masks: depth → list of masks
        self.depth_masks = {
            0: [dummy_mask()],
            1: [dummy_mask()],
        }


def test_create_segmentation_gif_to_buffer():
    image = dummy_image()

    seg_result = DummySegResult()
    buf = BytesIO()

    create_segmentation_gif(
        image,
        seg_result,
        io_buffer=buf,
        duration=100,
        final_duration=200,
        loop=0,
        highlight_idx=None,
    )

    # Buffer should now contain GIF data
    buf.seek(0)
    data = buf.read()
    assert len(data) > 0

    # Should be a valid GIF
    pil = Image.open(BytesIO(data))
    assert pil.format == "GIF"
