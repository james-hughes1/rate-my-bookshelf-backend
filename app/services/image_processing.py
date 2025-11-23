import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io


def read_image(image_path, max_dim):
    """Read and resize image."""
    image = cv2.imread(image_path)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = min(max_dim / h, max_dim / w, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w*scale), int(h*scale)), 
                          interpolation=cv2.INTER_AREA)
    return image


# ============================================================================
# Helper Functions
# ============================================================================

def compute_mask_bbox(mask, pad=0):
    """Get bounding box of mask with optional padding."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    y1 = max(int(ys.min()) - pad, 0)
    y2 = min(int(ys.max()) + pad, H - 1)
    x1 = max(int(xs.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad, W - 1)
    return (x1, y1, x2, y2)


def mask_to_bbox(mask):
    """Convert mask to simple axis-aligned bounding box (x1, y1, x2, y2)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def get_rotated_aspect_ratio(mask):
    """Get aspect ratio of minimum area rotated bounding box."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.inf
    pts = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    w, h = rect[1]
    if w == 0 or h == 0:
        return np.inf
    return max(w / h, h / w)


def get_longest_bbox_side(mask):
    """Get longest side of minimum area rotated bounding box."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0
    pts = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    w, h = rect[1]
    return max(w, h)


# ============================================================================
# Splitting Methods
# ============================================================================

def score_rect_split(image_gray, seg_bbox, pos, direction, center_penalty, 
                     soft_aspect_threshold):
    """Score a rectangular split using Sobel edge detection with penalties."""
    x1, y1, x2, y2 = seg_bbox
    band_width = 3
    width = x2 - x1
    height = y2 - y1
    
    if direction == 'vertical':
        x_start = max(x1, pos - band_width)
        x_end = min(x2, pos + band_width)
        band = image_gray[y1:y2, x_start:x_end]
        grad = cv2.Sobel(band, cv2.CV_64F, 1, 0, ksize=3)
        edge_score = np.sum(np.abs(grad)) / ((y2 - y1) ** 2)
        
        total_len = x2 - x1
        rel_pos = (pos - x1) / total_len
        cuts_shorter_side = width < height
    else:  # horizontal
        y_start = max(y1, pos - band_width)
        y_end = min(y2, pos + band_width)
        band = image_gray[y_start:y_end, x1:x2]
        grad = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)
        edge_score = np.sum(np.abs(grad)) / ((x2 - x1) ** 2)
        
        total_len = y2 - y1
        rel_pos = (pos - y1) / total_len
        cuts_shorter_side = height < width
    
    # Center penalty
    if cuts_shorter_side and max(width / height, height / width) > soft_aspect_threshold:
        penalty = 1.0
    else:
        dist_from_center = abs(rel_pos - 0.5) * 2
        penalty = (dist_from_center ** (1 + center_penalty))
    
    return penalty * edge_score


def find_best_rect_split(image_gray, seg_bbox, min_size, score_threshold, 
                         center_penalty, soft_aspect_threshold, hard_aspect_threshold):
    """Find best rectangular split for a bounding box."""
    x1, y1, x2, y2 = seg_bbox
    w, h = x2 - x1, y2 - y1
    
    if w < min_size * 2 and h < min_size * 2:
        return None
    
    # Check hard aspect ratio threshold
    if max(w / h, h / w) > hard_aspect_threshold:
        return None
    
    best_score = score_threshold
    best_split = None
    
    # Try vertical splits
    if w >= min_size * 2:
        for x in range(x1 + min_size, x2 - min_size, 5):
            score = score_rect_split(image_gray, seg_bbox, x, 'vertical',
                                    center_penalty, soft_aspect_threshold)
            if score > best_score:
                best_score = score
                best_split = ('vertical', x)
    
    # Try horizontal splits
    if h >= min_size * 2:
        for y in range(y1 + min_size, y2 - min_size, 5):
            score = score_rect_split(image_gray, seg_bbox, y, 'horizontal',
                                    center_penalty, soft_aspect_threshold)
            if score > best_score:
                best_score = score
                best_split = ('horizontal', y)
    
    return best_split


def split_rect_mask(mask, split_info):
    """Split a mask based on rectangular split."""
    direction, pos = split_info
    H, W = mask.shape
    
    if direction == 'vertical':
        child1 = mask.copy()
        child2 = mask.copy()
        child1[:, pos:] = 0
        child2[:, :pos] = 0
    else:  # horizontal
        child1 = mask.copy()
        child2 = mask.copy()
        child1[pos:, :] = 0
        child2[:pos, :] = 0
    
    return child1, child2


def extend_line_to_boundary(x1, y1, x2, y2, width, height):
    """Extend a line segment to image boundaries."""
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return x1, y1, x2, y2
    
    dx /= length
    dy /= length
    
    # Find all boundary intersections
    ts = []
    for X in [0, width - 1]:
        if dx != 0:
            t = (X - x1) / dx
            Y = y1 + t * dy
            if 0 <= Y < height:
                ts.append(t)
    for Y in [0, height - 1]:
        if dy != 0:
            t = (Y - y1) / dy
            X = x1 + t * dx
            if 0 <= X < width:
                ts.append(t)
    
    if len(ts) < 2:
        return x1, y1, x2, y2
    
    tmin, tmax = min(ts), max(ts)
    xA = x1 + tmin * dx
    yA = y1 + tmin * dy
    xB = x1 + tmax * dx
    yB = y1 + tmax * dy
    
    return int(xA), int(yA), int(xB), int(yB)


def find_best_hough_split(image_bgr, mask, pad, n_hough_lines, min_score, 
                          depth, ignore_min_score, min_pixels, min_side_fraction,
                          max_aspect_ratio):
    """Find best Hough line split for a mask."""
    H, W = image_bgr.shape[:2]
    bbox = compute_mask_bbox(mask, pad)
    if bbox is None:
        return None
    
    x1, y1, x2, y2 = bbox
    crop = image_bgr[y1:y2+1, x1:x2+1]
    crop_mask = mask[y1:y2+1, x1:x2+1].astype(np.uint8)
    Hc, Wc = crop_mask.shape
    
    # Compute parent variation
    parent_pixels = crop[crop_mask > 0]
    parent_variation = parent_pixels.std() if parent_pixels.size > 0 else 0
    
    # Edge detection + Hough
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    raw_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=40, minLineLength=0, maxLineGap=10)
    
    if raw_lines is None:
        return None
    
    # Score lines by detection fraction
    lines = []
    for (xa, ya, xb, yb) in raw_lines[:, 0]:
        dx, dy = xb - xa, yb - ya
        detected_len = np.hypot(dx, dy)
        if detected_len == 0:
            continue
        
        # Extend to boundaries
        xA, yA, xB, yB = extend_line_to_boundary(xa, ya, xb, yb, Wc, Hc)
        super_len = np.hypot(xB - xA, yB - yA)
        
        if super_len > 0:
            frac = detected_len / super_len
            # Convert to full image coordinates
            lines.append((frac, (xA + x1, yA + y1, xB + x1, yB + y1)))
    
    lines.sort(key=lambda x: x[0], reverse=True)
    top_lines = lines[:n_hough_lines]
    
    # Find best split by variation drop
    best_split = None
    best_score = 0
    
    xs_full, ys_full = np.meshgrid(np.arange(W), np.arange(H))
    
    for frac, (x1_line, y1_line, x2_line, y2_line) in top_lines:
        # Split mask by line
        lv = ((y2_line - y1_line) * (xs_full - x1_line) - 
              (x2_line - x1_line) * (ys_full - y1_line))
        child1 = mask.copy()
        child2 = mask.copy()
        child1[(lv < 0) | (mask == 0)] = 0
        child2[(lv >= 0) | (mask == 0)] = 0
        
        # --- Validate children immediately (like original) ---
        sum1, sum2 = child1.sum(), child2.sum()
        if sum1 < min_pixels or sum2 < min_pixels:
            continue
        
        # Check child bounding box sizes
        longest1 = get_longest_bbox_side(child1)
        longest2 = get_longest_bbox_side(child2)
        if (longest1 / min(H, W) < min_side_fraction or 
            longest2 / min(H, W) < min_side_fraction):
            continue
        
        # Check child aspect ratios
        if (get_rotated_aspect_ratio(child1) > max_aspect_ratio or 
            get_rotated_aspect_ratio(child2) > max_aspect_ratio):
            continue
        
        # --- Compute variation drop ---
        pixels1 = image_bgr[child1 > 0]
        pixels2 = image_bgr[child2 > 0]
        var1 = pixels1.std() if pixels1.size > 0 else 1e6
        var2 = pixels2.std() if pixels2.size > 0 else 1e6
        min_child_var = min(var1, var2)
        
        if parent_variation > 0:
            variation_drop = (parent_variation - min_child_var) / parent_variation
        else:
            variation_drop = 0
        
        # Orientation score (prefer axis-aligned)
        dx = x2_line - x1_line
        dy = y2_line - y1_line
        angle_folded = np.arctan2(dy, dx) % (np.pi / 2)
        orientation_score = 1 - np.sin(2 * angle_folded)
        
        overall_score = variation_drop * (orientation_score ** 2)
        
        if overall_score > best_score:
            if overall_score >= min_score or depth < ignore_min_score:
                best_score = overall_score
                best_split = (child1, child2, overall_score)
    
    return best_split


# ============================================================================
# Main Recursive Segmentation Function
# ============================================================================

def recursive_segment(image, mask=None, depth=0, method='rect', 
                     verbose=False, _split_count=[0], **params):
    """
    Recursively segment an image using either rectangular or Hough line splits.
    
    Parameters:
    -----------
    image : ndarray
        Input image (RGB or BGR)
    mask : ndarray, optional
        Binary mask defining current region (None = whole image)
    depth : int
        Current recursion depth
    method : str
        'rect' for rectangular Sobel splits, 'hough' for Hough line splits
    verbose : bool
        Print information each time a split is made
    **params : dict
        Method-specific parameters
        
    Returns:
    --------
    masks : list of ndarray
        List of binary masks for leaf segments
    """
    H, W = image.shape[:2]
    
    # Initialize mask if None
    if mask is None:
        mask = np.ones((H, W), dtype=np.uint8)
        _split_count[0] = 0  # Reset counter at root
    
    # Common stopping conditions
    max_depth = params.get('max_depth', 10)
    min_pixels = params.get('min_pixels', 200)
    
    if depth >= max_depth:
        return [mask]
    
    if mask.sum() < min_pixels:
        return []
    
    # Method-specific validation and splitting
    if method == 'rect':
        # Rectangular split method
        min_size_factor = params.get('min_size_factor', None)
        min_size = params.get('min_size', None)
        score_threshold = params.get('score_threshold', 0.2)
        center_penalty = params.get('center_penalty', 0.2)
        soft_aspect_threshold = params.get('soft_aspect_threshold', 3.0)
        hard_aspect_threshold = params.get('hard_aspect_threshold', 5.0)
        min_child_ratio = params.get('min_child_ratio', 0.3)
        
        # Compute min_size if not provided
        if min_size is None:
            if min_size_factor is None:
                min_size = min(H, W) // 20
            else:
                min_size = int(min(H, W) * min_size_factor)
        
        # Get bounding box
        seg_bbox = mask_to_bbox(mask)
        if seg_bbox is None:
            return []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Find best split
        split_info = find_best_rect_split(gray, seg_bbox, min_size, score_threshold,
                                         center_penalty, soft_aspect_threshold,
                                         hard_aspect_threshold)
        
        if split_info is None:
            return [mask]
        
        # Get score from split_info
        direction, pos = split_info
        best_score = score_rect_split(gray, seg_bbox, pos, direction,
                                      center_penalty, soft_aspect_threshold)
        
        # Split mask
        child1, child2 = split_rect_mask(mask, split_info)
        
        # Increment split counter and print if verbose
        if verbose:
            _split_count[0] += 1
            direction, pos = split_info
            print(f"Depth {depth} | Split {_split_count[0]}: "
                  f"score={best_score:.3f}, direction={direction}, pos={pos}, "
                  f"child ratios={child1.sum()/mask.sum():.2f}, "
                  f"{child2.sum()/mask.sum():.2f}")
        
        # Check min_child_ratio
        child1_w = (mask_to_bbox(child1)[2] - mask_to_bbox(child1)[0]) if mask_to_bbox(child1) else 0
        child1_h = (mask_to_bbox(child1)[3] - mask_to_bbox(child1)[1]) if mask_to_bbox(child1) else 0
        child2_w = (mask_to_bbox(child2)[2] - mask_to_bbox(child2)[0]) if mask_to_bbox(child2) else 0
        child2_h = (mask_to_bbox(child2)[3] - mask_to_bbox(child2)[1]) if mask_to_bbox(child2) else 0
        
        if not ((child1_w >= W * min_child_ratio or child1_h >= H * min_child_ratio) and
                (child2_w >= W * min_child_ratio or child2_h >= H * min_child_ratio)):
            return [mask]
        
    elif method == 'hough':
        # Hough line split method
        min_side_fraction = params.get('min_side_fraction', 0.3)
        max_aspect_ratio = params.get('max_aspect_ratio', 10)
        pad = params.get('pad', 5)
        n_hough_lines = params.get('n_hough_lines', 10)
        min_score = params.get('min_score', 0.01)
        ignore_min_score = params.get('ignore_min_score', 20)
        min_child_ratio = params.get('min_child_ratio', 0.2)
        
        # Check size constraints on parent
        longest_side = get_longest_bbox_side(mask)
        if longest_side / min(H, W) < min_side_fraction:
            return []
        
        # Check aspect ratio on parent
        aspect = get_rotated_aspect_ratio(mask)
        if aspect > max_aspect_ratio:
            return []
        
        # Find best split (child validation happens inside)
        split_result = find_best_hough_split(
            image, mask, pad, n_hough_lines, min_score, depth, ignore_min_score,
            min_pixels, min_side_fraction, max_aspect_ratio
        )
        
        if split_result is None:
            return [mask]
        
        child1, child2, score = split_result
        
        # Increment split counter and print if verbose
        if verbose:
            _split_count[0] += 1
            print(f"Depth {depth} | Split {_split_count[0]}: "
                  f"score={score:.3f}, "
                  f"child pixels={child1.sum()}, {child2.sum()}, "
                  f"child ratios={child1.sum()/mask.sum():.2f}, "
                  f"{child2.sum()/mask.sum():.2f}")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Validate children before recursing
    if child1.sum() < min_pixels or child2.sum() < min_pixels:
        return [mask]
    
    # Additional validation for rectangular method only
    # (Hough already validates children during split finding)
    if method == 'rect':
        # Check min_child_ratio for rect method
        child1_bbox = mask_to_bbox(child1)
        child2_bbox = mask_to_bbox(child2)
        if child1_bbox and child2_bbox:
            child1_w = child1_bbox[2] - child1_bbox[0]
            child1_h = child1_bbox[3] - child1_bbox[1]
            child2_w = child2_bbox[2] - child2_bbox[0]
            child2_h = child2_bbox[3] - child2_bbox[1]
            
            min_child_ratio = params.get('min_child_ratio', 0.3)
            if not ((child1_w >= W * min_child_ratio or child1_h >= H * min_child_ratio) and
                    (child2_w >= W * min_child_ratio or child2_h >= H * min_child_ratio)):
                return [mask]
    
    # Recurse on children
    masks1 = recursive_segment(image, child1, depth + 1, method, verbose, 
                               _split_count, **params)
    masks2 = recursive_segment(image, child2, depth + 1, method, verbose,
                               _split_count, **params)
    
    return masks1 + masks2


# ============================================================================
# GIF Creation
# ============================================================================

def draw_segment_boundaries(image, all_masks, boundary_color=(255, 165, 0), 
                            thickness=2):
    """
    Draw boundaries around all segments on the image.
    
    Parameters:
    -----------
    image : ndarray
        Original image
    all_masks : list of ndarray
        List of binary masks
    boundary_color : tuple
        RGB color for boundaries
    thickness : int
        Line thickness for boundaries
    
    Returns:
    --------
    ndarray : Image with boundaries drawn
    """
    result = image.copy()
    
    for mask in all_masks:
        # Find contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        cv2.drawContours(result, contours, -1, boundary_color, thickness)
    
    return result


def create_mean_value_image(image, masks):
    """
    Create an image where each segment is filled with its mean color.
    
    Parameters:
    -----------
    image : ndarray
        Original image
    masks : list of ndarray
        List of binary masks
    
    Returns:
    --------
    ndarray : Image with each segment filled with mean color
    """
    result = np.zeros_like(image)
    
    for mask in masks:
        if mask.sum() > 0:
            # Calculate mean color for this segment
            segment_pixels = image[mask > 0]
            mean_color = segment_pixels.mean(axis=0).astype(np.uint8)
            
            # Fill segment with mean color
            result[mask > 0] = mean_color
    
    return result


def _collect_masks_by_depth(image, mask=None, depth=0, method='rect', depth_masks=None, **params):
    """
    Recursively segment and collect masks at each depth level.
    
    Returns:
    --------
    depth_masks : dict
        Dictionary mapping depth -> list of masks at that depth
    """
    H, W = image.shape[:2]
    max_depth = params.get('max_depth', 10)
    
    if mask is None:
        mask = np.ones((H, W), dtype=np.uint8)
        depth_masks = {d: [] for d in range(max_depth + 1)}
    
    # Add current mask to this depth level
    depth_masks[depth].append(mask)
    
    # Stopping conditions
    min_pixels = params.get('min_pixels', 200)
    
    if depth >= max_depth:
        return depth_masks
    
    if mask.sum() < min_pixels:
        return depth_masks
    
    # Method-specific splitting logic
    if method == 'rect':
        min_size_factor = params.get('min_size_factor', None)
        min_size = params.get('min_size', None)
        score_threshold = params.get('score_threshold', 0.2)
        center_penalty = params.get('center_penalty', 0.2)
        soft_aspect_threshold = params.get('soft_aspect_threshold', 3.0)
        hard_aspect_threshold = params.get('hard_aspect_threshold', 5.0)
        
        if min_size is None:
            if min_size_factor is None:
                min_size = min(H, W) // 20
            else:
                min_size = int(min(H, W) * min_size_factor)
        
        seg_bbox = mask_to_bbox(mask)
        if seg_bbox is None:
            return depth_masks
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        split_info = find_best_rect_split(gray, seg_bbox, min_size, score_threshold,
                                         center_penalty, soft_aspect_threshold,
                                         hard_aspect_threshold)
        
        if split_info is None:
            return depth_masks
        
        child1, child2 = split_rect_mask(mask, split_info)
        
        # Validate children
        child1_bbox = mask_to_bbox(child1)
        child2_bbox = mask_to_bbox(child2)
        if child1_bbox and child2_bbox:
            child1_w = child1_bbox[2] - child1_bbox[0]
            child1_h = child1_bbox[3] - child1_bbox[1]
            child2_w = child2_bbox[2] - child2_bbox[0]
            child2_h = child2_bbox[3] - child2_bbox[1]
            
            min_child_ratio = params.get('min_child_ratio', 0.3)
            if not ((child1_w >= W * min_child_ratio or child1_h >= H * min_child_ratio) and
                    (child2_w >= W * min_child_ratio or child2_h >= H * min_child_ratio)):
                return depth_masks
    
    elif method == 'hough':
        min_side_fraction = params.get('min_side_fraction', 0.3)
        max_aspect_ratio = params.get('max_aspect_ratio', 10)
        pad = params.get('pad', 5)
        n_hough_lines = params.get('n_hough_lines', 10)
        min_score = params.get('min_score', 0.01)
        ignore_min_score = params.get('ignore_min_score', 20)
        
        longest_side = get_longest_bbox_side(mask)
        if longest_side / min(H, W) < min_side_fraction:
            return depth_masks
        
        aspect = get_rotated_aspect_ratio(mask)
        if aspect > max_aspect_ratio:
            return depth_masks
        
        split_result = find_best_hough_split(
            image, mask, pad, n_hough_lines, min_score, depth, ignore_min_score,
            min_pixels, min_side_fraction, max_aspect_ratio
        )
        
        if split_result is None:
            return depth_masks
        
        child1, child2, score = split_result
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Validate children
    if child1.sum() < min_pixels or child2.sum() < min_pixels:
        return depth_masks
    
    # Recurse on children
    depth_masks = _collect_masks_by_depth(image, child1, depth + 1, method, depth_masks, **params)
    depth_masks = _collect_masks_by_depth(image, child2, depth + 1, method, depth_masks, **params)
    
    return depth_masks


def create_segmentation_gif(image, output_path='segmentation.gif', method='rect',
                            duration=300, final_duration=2000, loop=0,
                            boundary_color=(255, 165, 0), thickness=2,
                            highlight_idx=None, io_buffer=None, **params):
    """
    Create an animated GIF showing the segmentation process depth by depth.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    output_path : str
        Path to save the GIF
    method : str
        'rect' or 'hough'
    duration : int
        Duration of each intermediate frame in milliseconds
    final_duration : int
        Duration of final mean-value frame in milliseconds
    loop : int
        Number of loops (0 = infinite)
    boundary_color : tuple
        RGB color for segment boundaries
    thickness : int
        Line thickness for boundaries
    highlight_idx : int
        Index of segment to highlight (default None)
    io_buffer : io.BytesIO() object
        To enable sending via FastAPI
    **params : dict
        Segmentation parameters (same as segment_rect or segment_hough)
    
    Returns:
    --------
    masks : list of binary masks (final leaf segments)
    """
    # Prepare parameters
    verbose = params.pop('verbose', False)
    max_depth = params.get('max_depth', 10)
    
    # Handle hough-specific preprocessing
    image_proc = image.copy()
    if method == 'hough':
        blur = params.pop('blur', True)
        blur_ksize = params.pop('blur_ksize', (7, 7))
        blur_sigma = params.pop('blur_sigma', 3)
        
        if blur:
            image_proc = cv2.GaussianBlur(image, blur_ksize, blur_sigma)
        
        if 'min_pixels' not in params or params['min_pixels'] is None:
            params['min_pixels'] = int(image.shape[0] * image.shape[1] * 0.01)
    
    # Collect masks at each depth
    depth_masks = _collect_masks_by_depth(image_proc, method=method, **params)
    
    # Create frames for each depth level
    gif_frames = []
    durations = []
    
    # Add original image as first frame
    gif_frames.append(image.copy())
    durations.append(duration)
    
    # Add frames for each depth with accumulating boundaries
    for depth in range(max_depth + 1):
        if len(depth_masks[depth]) > 0:
            # Draw boundaries for all segments up to this depth
            all_masks_so_far = []
            for d in range(depth + 1):
                all_masks_so_far.extend(depth_masks[d])
            
            frame = draw_segment_boundaries(image, all_masks_so_far, 
                                           boundary_color, thickness)
            gif_frames.append(frame)
            durations.append(duration)
            
            if verbose:
                print(f"Depth {depth}: {len(depth_masks[depth])} segments")
    
    # Get final leaf masks
    final_masks = []
    for depth in range(max_depth, -1, -1):
        if len(depth_masks[depth]) > 0:
            final_masks = depth_masks[depth]
            break
    
    # Add final frame with mean values
    if highlight_idx is None:
        final_frame = create_mean_value_image(image, all_masks_so_far)
    else:
        final_frame = visualize_selected_segment(image, all_masks_so_far, highlight_idx)
        
    gif_frames.append(final_frame)
    durations.append(final_duration)
    
    # Convert frames to PIL Images
    pil_frames = []
    for frame in gif_frames:
        if frame.shape[2] == 3:
            pil_frame = Image.fromarray(frame)
        else:
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_frames.append(pil_frame)
    
    # Save as GIF with varying durations
    if len(pil_frames) > 0:
        if output_path is not None:
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=durations,
                loop=loop
            )
        elif io_buffer is not None:
            pil_frames[0].save(
                io_buffer,
                format="GIF",
                save_all=True,
                append_images=pil_frames[1:],
                duration=durations,
                loop=loop
            )
        print(f"\nGIF saved to {output_path} ({len(pil_frames)} frames)")
    
    print(f"Total leaf segments: {len(final_masks)}")
    return final_masks


# ============================================================================
# Convenience Functions
# ============================================================================

def segment_rect(image, min_size_factor=None, min_size=None, 
                 center_penalty=0.2, soft_aspect_threshold=3.0,
                 hard_aspect_threshold=5.0, score_threshold=0.2,
                 min_child_ratio=0.3, max_depth=10, min_pixels=200,
                 verbose=False):
    """
    Segment image using rectangular Sobel-based splits.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    min_size_factor : float, optional
        Minimum size as fraction of image dimension
    min_size : int, optional
        Minimum size in pixels (overrides min_size_factor if provided)
    center_penalty : float
        Penalty factor for splits near center
    soft_aspect_threshold : float
        Aspect ratio threshold for applying center penalty softly
    hard_aspect_threshold : float
        Aspect ratio beyond which splits are not allowed
    score_threshold : float
        Minimum score required to accept a split
    min_child_ratio : float
        Minimum ratio of child segment size to original image size
    max_depth : int
        Maximum recursion depth
    min_pixels : int
        Minimum pixels in a segment
    verbose : bool
        Print info each time a split is made
    
    Returns:
    --------
    masks : list of binary masks
    """
    params = {
        'min_size_factor': min_size_factor,
        'min_size': min_size,
        'center_penalty': center_penalty,
        'soft_aspect_threshold': soft_aspect_threshold,
        'hard_aspect_threshold': hard_aspect_threshold,
        'score_threshold': score_threshold,
        'min_child_ratio': min_child_ratio,
        'max_depth': max_depth,
        'min_pixels': min_pixels
    }
    
    masks = recursive_segment(image, method='rect', verbose=verbose, **params)
    print(f"\nTotal segments: {len(masks)}")
    return masks


def segment_hough(image, max_depth=10, min_pixels=None, min_side_fraction=0.3,
                  pad=5, min_child_ratio=0.2, min_score=0.05, 
                  n_hough_lines=10, max_aspect_ratio=10,
                  blur=True, blur_ksize=(7, 7), blur_sigma=3, 
                  ignore_min_score=5, verbose=False):
    """
    Segment image using Hough line splits with variation scoring.
    
    Parameters:
    -----------
    image : ndarray
        Input image
    max_depth : int
        Maximum recursion depth
    min_pixels : int, optional
        Minimum pixels in a segment (default: 1% of image area)
    min_side_fraction : float
        Minimum fraction of shortest image side for bbox longest side
    pad : int
        Padding around bounding box for Hough detection
    min_child_ratio : float
        Minimum ratio of child to parent pixels
    min_score : float
        Minimum score to accept a split
    n_hough_lines : int
        Number of top candidate lines to score
    max_aspect_ratio : float
        Maximum allowed aspect ratio of rotated bbox
    blur : bool
        Whether to apply Gaussian blur
    blur_ksize : tuple
        Gaussian blur kernel size
    blur_sigma : float
        Gaussian blur sigma
    ignore_min_score : int
        Number of initial splits to accept regardless of min_score
    verbose : bool
        Print info each time a split is made
    
    Returns:
    --------
    masks : list of binary masks
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if blur:
        image = cv2.GaussianBlur(image, blur_ksize, blur_sigma)
    
    # Default min_pixels to 1% of image area
    if min_pixels is None:
        min_pixels = int(image.shape[0] * image.shape[1] * 0.01)
    
    params = {
        'max_depth': max_depth,
        'min_pixels': min_pixels,
        'min_side_fraction': min_side_fraction,
        'pad': pad,
        'min_child_ratio': min_child_ratio,
        'min_score': min_score,
        'n_hough_lines': n_hough_lines,
        'max_aspect_ratio': max_aspect_ratio,
        'ignore_min_score': ignore_min_score,
    }
    
    masks = recursive_segment(image, method='hough', verbose=verbose, **params)
    print(f"\nTotal segments: {len(masks)}")
    return masks


# ============================================================================
# Visualization
# ============================================================================

def visualize_masks(image, masks, max_show=10):
    """Visualize segmentation masks."""
    n = min(len(masks), max_show)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i, mask in enumerate(masks[:n]):
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        
        # Show mask
        axes[i, 0].imshow(mask, cmap='gray')
        axes[i, 0].set_title(f'Mask {i+1} ({mask.sum()} pixels)')
        axes[i, 0].axis('off')
        
        # Show cropped region
        crop = image[y1:y2, x1:x2]
        axes[i, 1].imshow(crop)
        axes[i, 1].set_title(f'Crop {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_selected_segment(image, masks, selected_mask_idx, 
                               highlight_color=(255, 165, 0), thickness=4, 
                               dash_length=5):
    """
    Create mean-valued segmentation image and highlight selected segment.

    Args:
        image: Original image (H,W,3)
        masks: list of binary masks
        selected_mask_idx: index of mask to highlight
        highlight_color: RGB color for highlight
        thickness: line thickness
        dash_length: length of dashes in dashed line

    Returns:
        vis_img: mean-valued image with dashed box around selected segment
    """
    # Create mean value image
    mean_img = create_mean_value_image(image, masks)
    
    # Get selected mask and its bounding box
    if selected_mask_idx >= len(masks):
        return mean_img
    
    selected_mask = masks[selected_mask_idx]
    bbox = mask_to_bbox(selected_mask)
    
    if bbox is None:
        return mean_img
    
    x1, y1, x2, y2 = bbox
    
    # Draw dashed box around selected segment
    for i in range(x1, x2, dash_length * 2):
        cv2.line(mean_img, (i, y1), (min(i + dash_length, x2), y1), 
                highlight_color, thickness)
        cv2.line(mean_img, (i, y2), (min(i + dash_length, x2), y2), 
                highlight_color, thickness)
    
    for i in range(y1, y2, dash_length * 2):
        cv2.line(mean_img, (x1, i), (x1, min(i + dash_length, y2)), 
                highlight_color, thickness)
        cv2.line(mean_img, (x2, i), (x2, min(i + dash_length, y2)), 
                highlight_color, thickness)
    
    return mean_img
