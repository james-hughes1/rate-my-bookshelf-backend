import numpy as np

from app.services.image_processing import (
    extend_line_to_boundary,
    get_longest_bbox_side,
    get_rotated_aspect_ratio,
    mask_to_bbox,
    split_rect_mask,
)


def test_mask_to_bbox_normal():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1

    bbox = mask_to_bbox(mask)
    assert bbox == (3, 2, 6, 4)  # x1, y1, x2, y2


def test_mask_to_bbox_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert mask_to_bbox(mask) is None


def test_mask_to_bbox_with_padding():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 1

    bbox = mask_to_bbox(mask, pad=2)
    assert bbox == (2, 2, 7, 7)


def test_get_rotated_aspect_ratio_square():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:7, 2:7] = 1

    ar = get_rotated_aspect_ratio(mask)
    assert 0.9 < ar < 1.1  # approx square


def test_get_rotated_aspect_ratio_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert get_rotated_aspect_ratio(mask) == float("inf")


def test_get_longest_bbox_side():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:16, 10:21] = 1

    longest = get_longest_bbox_side(mask)
    assert longest == 10  # height = 10, width = 10


def test_get_longest_bbox_side_empty():
    mask = np.zeros((20, 30), dtype=np.uint8)
    assert get_longest_bbox_side(mask) == 0


def test_split_rect_mask_vertical():
    mask = np.ones((10, 10), dtype=np.uint8)

    child1, child2 = split_rect_mask(mask, ("vertical", 5))

    assert child1[:, 5:].sum() == 0
    assert child2[:, :5].sum() == 0
    assert child1.sum() + child2.sum() == mask.sum()


def test_split_rect_mask_horizontal():
    mask = np.ones((10, 10), dtype=np.uint8)

    child1, child2 = split_rect_mask(mask, ("horizontal", 6))

    assert child1[6:, :].sum() == 0
    assert child2[:6, :].sum() == 0


def test_extend_line_to_boundary_diagonal():
    x1, y1, x2, y2 = 2, 2, 8, 8
    width, height = 10, 10

    Xa, Ya, Xb, Yb = extend_line_to_boundary(x1, y1, x2, y2, width, height)

    # Should hit corners (0,0) and (9,9)
    assert (Xa, Ya) == (0, 0)
    assert (Xb, Yb) == (9, 9)


def test_extend_line_to_boundary_zero_length():
    Xa, Ya, Xb, Yb = extend_line_to_boundary(5, 5, 5, 5, 10, 10)
    assert (Xa, Ya, Xb, Yb) == (5, 5, 5, 5)
