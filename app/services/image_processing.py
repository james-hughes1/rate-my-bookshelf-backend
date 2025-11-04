import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, image_path, min_size=None,
                 center_penalty=0.2, soft_aspect_threshold=3.0,
                 hard_aspect_threshold=5.0, score_threshold=0.2,
                 min_child_ratio=0.3, max_dim=1024):
        """
        Initialize the SimpleSegmenter with the given parameters.
        """
        image = cv2.imread(image_path)
        if image.shape[2] == 3:
            # Convert BGR → RGB if it looks like BGR (OpenCV default)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        scale = min(max_dim / h, max_dim / w, 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if min_size is None:
            self.min_size = min(image.shape[:2]) // 20
        else:
            self.min_size = min_size
        self.center_penalty = center_penalty
        self.soft_aspect_threshold = soft_aspect_threshold
        self.hard_aspect_threshold = hard_aspect_threshold
        self.score_threshold = score_threshold
        self.min_child_ratio = min_child_ratio

        # Confidence dictionary
        self.segment_confidence = {}

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
