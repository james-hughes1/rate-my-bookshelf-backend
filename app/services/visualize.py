import cv2
import numpy as np
from PIL import Image
from .image_processing import mask_to_bbox

def draw_segment_boundaries(image, all_masks, boundary_color=(255, 165, 0), 
                            thickness=2):
    """Draw boundaries around all segments on the image."""
    result = image.copy()
    
    for mask in all_masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, boundary_color, thickness)
    
    return result


def create_mean_value_image(image, masks):
    """
    Create an image where each segment is filled with its mean color.
    
    Parameters:
    -----------
    image : ndarray
        Original image (H, W, 3)
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


def visualize_selected_segment(
        image, masks, selected_mask_idx,
        highlight_color=(255, 165, 0),
        mask_tint_color=(60, 60, 60),     # darker shade applied to mask interior
        mask_tint_alpha=0.4,             # 0 = no tint, 1 = full tint
        thickness=3,
        dash_length=6):
    """
    Visualize segmentation with a dashed contour tightly around the mask shape
    and optionally tint the mask interior.

    Args:
        image: Original image (H,W,3)
        masks: list of binary masks
        selected_mask_idx: index of mask to highlight
        highlight_color: BGR/RGB dashed outline color
        mask_tint_color: color blended into the mask interior
        mask_tint_alpha: blend amount [0-1]
        thickness: thickness of dashed outline
        dash_length: dash length for dashed contour

    Returns:
        vis_img: mean-value image with contour highlight
    """

    # Step 1 — mean-filled background
    vis_img = create_mean_value_image(image, masks)

    if selected_mask_idx >= len(masks):
        return vis_img

    selected_mask = masks[selected_mask_idx].astype(np.uint8)

    # Step 2 — optional interior tint
    if mask_tint_alpha > 0:
        tint = np.full_like(vis_img, mask_tint_color, dtype=np.uint8)
        mask_3c = np.stack([selected_mask]*3, axis=-1)
        vis_img = np.where(mask_3c == 1,
                           (vis_img * (1 - mask_tint_alpha) + tint * mask_tint_alpha).astype(np.uint8),
                           vis_img)

    # Step 3 — find tight contour(s)
    contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return vis_img

    # Step 4 — draw dashed contour
    for contour in contours:
        contour = contour.squeeze()
        if len(contour.shape) != 2:
            continue

        # walk along contour as polyline and draw dashed segments
        for i in range(len(contour)):
            p1 = tuple(contour[i])
            p2 = tuple(contour[(i + 1) % len(contour)])

            # get segment vector length
            seg_len = int(np.linalg.norm(np.array(p2) - np.array(p1)))
            if seg_len == 0:
                continue

            # interpolate points along segment
            for offset in range(0, seg_len, dash_length * 2):
                start_t = offset / seg_len
                end_t   = min(offset + dash_length, seg_len) / seg_len

                s = (int(p1[0] + (p2[0] - p1[0]) * start_t),
                     int(p1[1] + (p2[1] - p1[1]) * start_t))
                e = (int(p1[0] + (p2[0] - p1[0]) * end_t),
                     int(p1[1] + (p2[1] - p1[1]) * end_t))

                cv2.line(vis_img, s, e, highlight_color, thickness)

    return vis_img


def create_segmentation_gif(image, segmentation_result, output_path=None,
                            duration=300, final_duration=2000, loop=0,
                            boundary_color=(255, 165, 0), thickness=2,
                            highlight_idx=None, io_buffer=None):
    """
    Create an animated GIF from segmentation result.
    
    Parameters:
    -----------
    image : ndarray
        Original image
    segmentation_result : SegmentationResult
        Result from segment_with_tree()
    output_path : str, optional
        Path to save GIF (if None, must provide io_buffer)
    duration : int
        Duration of intermediate frames in milliseconds
    final_duration : int
        Duration of final frame in milliseconds
    loop : int
        Number of loops (0 = infinite)
    boundary_color : tuple
        RGB color for boundaries
    thickness : int
        Line thickness
    highlight_idx : int, optional
        Index of segment to highlight in final frame
    io_buffer : io.BytesIO, optional
        Buffer to write GIF to (for FastAPI)
    """
    depth_masks = segmentation_result.depth_masks
    max_depth = max(depth_masks.keys())
    
    # Create frames
    gif_frames = []
    durations = []
    
    # Original image
    gif_frames.append(image.copy())
    durations.append(duration)
    
    # Add frames for each depth with accumulating boundaries
    for depth in range(max_depth + 1):
        if len(depth_masks[depth]) > 0:
            all_masks_so_far = []
            for d in range(depth + 1):
                all_masks_so_far.extend(depth_masks[d])
            
            frame = draw_segment_boundaries(image, all_masks_so_far, 
                                           boundary_color, thickness)
            gif_frames.append(frame)
            durations.append(duration)
    
    # Final frame with mean values
    if highlight_idx is None:
        final_frame = create_mean_value_image(image, segmentation_result.masks)
    else:
        final_frame = visualize_selected_segment(
            image,
            segmentation_result.masks,
            highlight_idx,
            highlight_color=boundary_color, 
            mask_tint_color=boundary_color,
            mask_tint_alpha=0.4,
            thickness=3,
            dash_length=6
        )
    
    gif_frames.append(final_frame)
    durations.append(final_duration)
    
    # Convert to PIL
    pil_frames = []
    for frame in gif_frames:
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            pil_frame = Image.fromarray(frame)
        else:
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_frames.append(pil_frame)
    
    # Save GIF
    if len(pil_frames) > 0:
        save_kwargs = {
            'save_all': True,
            'append_images': pil_frames[1:],
            'duration': durations,
            'loop': loop
        }
        
        if output_path is not None:
            pil_frames[0].save(output_path, **save_kwargs)
            print(f"GIF saved to {output_path} ({len(pil_frames)} frames)")
        elif io_buffer is not None:
            pil_frames[0].save(io_buffer, format='GIF', **save_kwargs)
            print(f"GIF written to buffer ({len(pil_frames)} frames)")

