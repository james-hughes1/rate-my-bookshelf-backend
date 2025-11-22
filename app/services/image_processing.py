import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(image_path, max_dim):
    image = cv2.imread(image_path)
    if image.shape[2] == 3:
        # Convert BGR → RGB if it looks like BGR (OpenCV default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = min(max_dim / h, max_dim / w, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return image

class SimpleSegmenter:
    """
    A simple image segmenter that recursively splits an image into segments

    Arguments:
        image_path: Path to the input image.
        min_size: Minimum size (in pixels) for a segment to be considered for splitting.
        center_penalty: Penalty factor for splits near the center.
        soft_aspect_threshold: Aspect ratio threshold for applying center penalty softly.
        hard_aspect_threshold: Aspect ratio threshold beyond which splits are not allowed.
        score_threshold: Minimum score required to accept a split.
        min_child_ratio: Minimum ratio of child segment size to original image size to keep a segment.
    
    Methods:
        segment(): Perform segmentation and return list of segments.
        try_split(seg): Attempt to split a segment and return children if successful.
        score_split(seg, pos, direction, center_penalty): Score a potential split.
        visualize_segments(segments, max_show=10): Visualize the segments.
        get_crops(segments): Return cropped images and their confidence scores.
    """
    def __init__(self, min_size_factor=None,
                 center_penalty=0.2, soft_aspect_threshold=3.0,
                 hard_aspect_threshold=5.0, score_threshold=0.2,
                 min_child_ratio=0.3):
        """
        Initialize the SimpleSegmenter with the given parameters.
        """
        self.min_size_factor = min_size_factor
        self.center_penalty = center_penalty
        self.soft_aspect_threshold = soft_aspect_threshold
        self.hard_aspect_threshold = hard_aspect_threshold
        self.score_threshold = score_threshold
        self.min_child_ratio = min_child_ratio

        # Confidence dictionary
        self.segment_confidence = {}

    def load(self, image, max_dim=1024):
        h, w = image.shape[:2]
        scale = min(max_dim / h, max_dim / w, 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        self.image = image
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if self.min_size_factor is None:
            self.min_size = min(self.image.shape[:2]) // 20
        else:
            self.min_size = int(min(self.image.shape[:2]) * self.min_size_factor)

    def segment(self):
        """
        Perform segmentation on the image and return list of segments.
        Each segment is represented as (x1, y1, x2, y2).

        Returns:
            List of segments.
        """
        h, w = self.image.shape[:2]
        segments = [(0, 0, w, h)]
        self.segment_confidence[(0, 0, w, h)] = 1.0

        changed = True
        while changed:
            changed = False
            new_segments = []

            for seg in segments:
                split_result = self.try_split(seg)
                if split_result:
                    children, confidence = split_result
                    for child in children:
                        child_w = child[2] - child[0]
                        child_h = child[3] - child[1]
                        # only include if above min_child_ratio threshold
                        if (child_w >= w * self.min_child_ratio or
                            child_h >= h * self.min_child_ratio):
                            self.segment_confidence[child] = confidence
                    new_segments.extend(children)
                    changed = True
                else:
                    new_segments.append(seg)

            segments = new_segments
            print(f"Now have {len(segments)} segments")

        # Sort leaves by area
        sorted_segments = sorted(
            segments,
            key=lambda s: (s[2] - s[0]) * (s[3] - s[1]),
            reverse=True
        )

        print(f"\nFound {len(sorted_segments)} leaf segments.")
        return sorted_segments

    def try_split(self, seg):
        """
        Attempt to split a segment; return children and score if successful.

        Args:
            seg: (x1, y1, x2, y2) defining the segment

        Returns:
            (children, score) if split is successful, else None
        """
        x1, y1, x2, y2 = seg
        width, height = x2 - x1, y2 - y1

        if width < self.min_size * 2 or height < self.min_size * 2:
            return None
        if max(width / height, height / width) > self.hard_aspect_threshold:
            return None

        best_split = None
        best_score = self.score_threshold

        # Vertical
        if width >= self.min_size * 2:
            for x in range(x1 + self.min_size, x2 - self.min_size, 5):
                score = self.score_split(seg, x, 'vertical', self.center_penalty)
                if score > best_score:
                    best_score = score
                    best_split = ('vertical', x)

        # Horizontal
        if height >= self.min_size * 2:
            for y in range(y1 + self.min_size, y2 - self.min_size, 5):
                score = self.score_split(seg, y, 'horizontal', self.center_penalty)
                if score > best_score:
                    best_score = score
                    best_split = ('horizontal', y)

        if best_split:
            direction, pos = best_split
            if direction == 'vertical':
                children = [(x1, y1, pos, y2), (pos, y1, x2, y2)]
            else:
                children = [(x1, y1, x2, pos), (x1, pos, x2, y2)]
            return children, best_score

        return None

    def score_split(self, seg, pos, direction, center_penalty):
        """
        Score a potential split at position `pos` in `direction` for segment `seg`.

        Args:
            seg: (x1, y1, x2, y2) defining the segment
            pos: Position to split
            direction: 'vertical' or 'horizontal'
            center_penalty: Penalty factor for center splits

        Returns:
            float: Score of the split
        """
        x1, y1, x2, y2 = seg
        band_width = 3
        width = x2 - x1
        height = y2 - y1

        if direction == 'vertical':
            split_band = self.gray[y1:y2, max(x1, pos - band_width):min(x2, pos + band_width)]
            grad = cv2.Sobel(split_band, cv2.CV_64F, 1, 0, ksize=3)
            total_len = x2 - x1
            rel_pos = (pos - x1) / total_len
            cuts_shorter_side = width < height

            # Edge score
            edge_score = np.sum(np.abs(grad)) / ((y2 - y1)**2)
        else:
            split_band = self.gray[max(y1, pos - band_width):min(y2, pos + band_width), x1:x2]
            grad = cv2.Sobel(split_band, cv2.CV_64F, 0, 1, ksize=3)
            total_len = y2 - y1
            rel_pos = (pos - y1) / total_len
            cuts_shorter_side = height < width

            # Edge score
            edge_score = np.sum(np.abs(grad)) / ((x2 - x1)**2)

        # Center penalty
        if cuts_shorter_side and max(width / height, height / width) > self.soft_aspect_threshold:
            penalty = 1.0
        else:
            dist_from_center = abs(rel_pos - 0.5) * 2
            penalty = (dist_from_center ** (1 + center_penalty))

        return penalty * edge_score

    def visualize_segments(self, segments, max_show=10):
        """
        Visualize the segments on the image.

        Args:
            segments: List of segments to visualize.
            max_show: Maximum number of segments to show.
        """
        n = min(len(segments), max_show)
        plt.figure(figsize=(12, 3 * n))
        for i, seg in enumerate(segments[:n]):
            x1, y1, x2, y2 = seg
            crop = self.image[y1:y2, x1:x2]
            plt.subplot(n, 1, i + 1)
            plt.imshow(crop)
            plt.title(f"Segment {i+1} | Confidence = {self.segment_confidence.get(seg, 0):.3f}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def get_crops(self, segments):
        """
        Return cropped images and their confidence scores.

        Args:
            segments: List of segments to crop.

        Returns:
            List of tuples (crop, confidence, (x1, y1, x2, y2)).
        """
        crops = []
        for seg in segments:
            x1, y1, x2, y2 = seg
            crop = self.image[y1:y2, x1:x2].copy()
            conf = self.segment_confidence.get(seg, 0)
            crops.append((crop, conf, seg))
        return crops


def mean_value_spine_image(img, spines):
    """
    Produce a mean-valued version of the image based on spine bounding boxes.

    Args:
        img: Original image (H,W,3)
        spines: list of 4-tuples [(x1,y1,x2,y2), ...]

    Returns:
        mean_img: Image where each spine segment is filled with its mean color
    """
    mean_img = img.copy()

    for x1, y1, x2, y2 in spines:
        seg = img[y1:y2, x1:x2]
        mean_val = seg.mean(axis=(0,1)).astype(np.uint8)
        mean_img[y1:y2, x1:x2] = mean_val

    return mean_img


def visualize_selected_segments(img, selected_segments, color = (255, 165, 0), thickness = 4, dash_length = 5):
    """
    Create mean-valued segmentation image and highlight selected segments.

    Args:
        img: Original image (H,W,3)
        selected_segments: list of tuples [(string, [x1,y1,x2,y2]), ...]

    Returns:
        vis_img: mean-valued image with orange dashed boxes around selected segments
    """
    for string, (x1, y1, x2, y2) in selected_segments:
        seg = img[y1:y2, x1:x2]

        # Draw orange dashed box
        for i in range(x1, x2, dash_length*2):
            cv2.line(img, (i, y1), (min(i+dash_length, x2), y1), color, thickness)
            cv2.line(img, (i, y2), (min(i+dash_length, x2), y2), color, thickness)
        for i in range(y1, y2, dash_length*2):
            cv2.line(img, (x1, i), (x1, min(i+dash_length, y2)), color, thickness)
            cv2.line(img, (x2, i), (x2, min(i+dash_length, y2)), color, thickness)

    return img


def thd_split_mask_variation(img_bgr, mask=None, depth=0, max_depth=6,
                             min_pixels=200, min_side_fraction=0.2, pad=5,
                             min_child_ratio=0.05, min_score=0.2,
                             n_hough_lines=5, lines_so_far=None,
                             max_aspect_ratio=10,
                             blur=True, blur_ksize=(7,7), blur_sigma=3, ignore_min_score=1):
    """
    Recursive tree-based segmentation using Hough lines scored by variation drop,
    with max_aspect_ratio filter using rotated bounding rectangles, and
    stopping condition that prevents creation of too-small segments.

    Parameters
    ----------
    img_bgr : np.ndarray
        Original image (HxWx3, BGR)
    mask : np.ndarray
        Current mask to split (HxW)
    depth : int
        Current recursion depth
    max_depth : int
        Maximum recursion depth
    min_pixels : int
        Minimum mask pixels to continue splitting / create segment
    min_side_fraction : float
        Minimum fraction of the shortest image side that the longest bounding box side must have
    pad : int
        Pixels to pad around bounding box when cropping (for Hough line detection)
    min_child_ratio : float
        Minimum fraction of parent pixels each child must have
    min_score : float
        Minimum score to accept a split
    n_hough_lines : int
        Number of top candidate lines by superline fraction to score
    lines_so_far : list
        Recorded split lines
    max_aspect_ratio : float
        Maximum allowed aspect ratio of rotated bounding box
    blur : bool
        Whether to apply Gaussian blur at top level
    blur_ksize : tuple
        Gaussian blur kernel size
    blur_sigma : float
        Gaussian blur sigma
    """
    if lines_so_far is None:
        lines_so_far = []

    if depth == 0 and blur:
        img_bgr = cv2.GaussianBlur(img_bgr, blur_ksize, blur_sigma)
    
    H, W = img_bgr.shape[:2]

    if mask is None:
        mask = np.ones((H,W), dtype=np.uint8)

    # --- Reject mask immediately if too small ---
    if mask.sum() < min_pixels:
        return [], lines_so_far

    # Reject mask if bounding box longest side is too small
    ys, xs = np.nonzero(mask)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box_w, box_h = rect[1]
    if box_w == 0 or box_h == 0:
        return [], lines_so_far
    longest_side = max(box_w, box_h)
    if longest_side / min(H, W) < min_side_fraction:
        return [], lines_so_far

    if depth >= max_depth:
        return [mask], lines_so_far

    # --- Crop bounding box for Hough line detection ---
    y1, y2 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, H-1)
    x1, x2 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, W-1)
    crop = img_bgr[y1:y2+1, x1:x2+1]
    crop_mask = mask[y1:y2+1, x1:x2+1].astype(np.uint8)
    Hc, Wc = crop_mask.shape

    parent_pixels = crop[crop_mask > 0]
    parent_variation = parent_pixels.std() if parent_pixels.size > 0 else 0

    # --- Edge detection + Hough ---
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    raw_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=40, minLineLength=0, maxLineGap=10)
    if raw_lines is None:
        return [mask], lines_so_far

    # --- Compute superline fraction ---
    def compute_fraction_line_crop(xa, ya, xb, yb, absolute=False):
        dx, dy = xb - xa, yb - ya
        detected_len = np.hypot(dx, dy)
        if detected_len == 0: return 0, (xa, ya, xb, yb)
        dx /= detected_len; dy /= detected_len
        ts = []
        for X in [0, Wc-1]:
            if dx != 0:
                t = (X - xa)/dx
                Y = ya + t*dy
                if 0 <= Y < Hc: ts.append(t)
        for Y in [0, Hc-1]:
            if dy != 0:
                t = (Y - ya)/dy
                X = xa + t*dx
                if 0 <= X < Wc: ts.append(t)
        if len(ts) < 2: return 0, (xa, ya, xb, yb)
        tmin, tmax = min(ts), max(ts)
        xA, yA = xa + tmin*dx, ya + tmin*dy
        xB, yB = xa + tmax*dx, ya + tmax*dy
        super_len = np.hypot(xB-xA, yB-yA)
        if super_len == 0: return 0, (xa, ya, xb, yb)
        if not absolute:
            return detected_len / super_len, (xA, yA, xB, yB)
        else:
            return detected_len, (xA, yA, xB, yB)

    # --- Top candidate lines ---
    lines = []
    for (xa, ya, xb, yb) in raw_lines[:,0]:
        frac, (xA, yA, xB, yB) = compute_fraction_line_crop(xa, ya, xb, yb, absolute=False)
        lines.append((frac, (int(xA)+x1, int(yA)+y1, int(xB)+x1, int(yB)+y1)))
    lines.sort(key=lambda x: x[0], reverse=True)
    top_lines = lines[:n_hough_lines]

    best_split = None
    best_score = 0

    xs_full, ys_full = np.meshgrid(np.arange(W), np.arange(H))
    for frac, (x1_line, y1_line, x2_line, y2_line) in top_lines:
        lv = (y2_line - y1_line)*(xs_full - x1_line) - (x2_line - x1_line)*(ys_full - y1_line)
        child1 = mask.copy(); child2 = mask.copy()
        child1[(lv < 0) | (mask==0)] = 0
        child2[(lv >= 0) | (mask==0)] = 0

        # --- Reject children immediately if too small ---
        sum1, sum2 = child1.sum(), child2.sum()
        if sum1 < min_pixels or sum2 < min_pixels:
            continue

        # Reject children if bounding box longest side too small
        def child_ok(m):
            ys, xs = np.nonzero(m)
            if len(xs) == 0: return False
            rect = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
            w, h = rect[1]
            return max(w,h)/min(H,W) >= min_side_fraction
        if not child_ok(child1) or not child_ok(child2):
            continue

        # --- Max aspect ratio check ---
        def max_rotated_aspect_ratio(m):
            ys, xs = np.nonzero(m)
            if len(xs) == 0: return np.inf
            rect = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
            w, h = rect[1]
            if w == 0 or h == 0: return np.inf
            return max(w/h, h/w)
        if max_rotated_aspect_ratio(child1) > max_aspect_ratio or max_rotated_aspect_ratio(child2) > max_aspect_ratio:
            continue

        # --- Compute variation drop ---
        pixels1 = img_bgr[child1>0]; pixels2 = img_bgr[child2>0]
        var1 = pixels1.std() if pixels1.size>0 else 1e6
        var2 = pixels2.std() if pixels2.size>0 else 1e6
        min_child_var = min(var1, var2)
        variation_drop = (parent_variation - min_child_var)/parent_variation if parent_variation>0 else 0

        # --- Orientation score ---
        dx = x2_line - x1_line; dy = y2_line - y1_line
        angle_folded = np.arctan2(dy, dx) % (np.pi/2)
        orientation_score = (1 - np.sin(2 * angle_folded))
        overall_score = variation_drop * (orientation_score**2)

        if overall_score > best_score and (overall_score >= min_score or len(lines_so_far) < ignore_min_score):
            best_score = overall_score
            best_split = (x1_line, y1_line, x2_line, y2_line, child1, child2)

    if best_split is None:
        return [mask], lines_so_far

    x1_line, y1_line, x2_line, y2_line, child1, child2 = best_split
    lines_so_far.append((x1_line, y1_line, x2_line, y2_line))
    print(f"Depth {depth} | Segments {len(lines_so_far)}: best score={best_score:.3f}, "
          f"child ratios={child1.sum()/mask.sum():.2f}, {child2.sum()/mask.sum():.2f}")

    leaves1, lines_so_far = thd_split_mask_variation(img_bgr, child1, depth+1, max_depth,
                                                     min_pixels=min_pixels,
                                                     min_side_fraction=min_side_fraction,
                                                     pad=pad, min_child_ratio=min_child_ratio,
                                                     min_score=min_score,
                                                     n_hough_lines=n_hough_lines,
                                                     lines_so_far=lines_so_far,
                                                     max_aspect_ratio=max_aspect_ratio,
                                                     ignore_min_score=ignore_min_score)
    leaves2, lines_so_far = thd_split_mask_variation(img_bgr, child2, depth+1, max_depth,
                                                     min_pixels=min_pixels,
                                                     min_side_fraction=min_side_fraction,
                                                     pad=pad, min_child_ratio=min_child_ratio,
                                                     min_score=min_score,
                                                     n_hough_lines=n_hough_lines,
                                                     lines_so_far=lines_so_far,
                                                     max_aspect_ratio=max_aspect_ratio,
                                                     ignore_min_score=ignore_min_score)

    all_leaves = [m for m in leaves1 + leaves2 if m.sum() > 0]
    if depth == 0:
        print(f"Total non-empty segments: {len(all_leaves)}")
    return all_leaves, lines_so_far
