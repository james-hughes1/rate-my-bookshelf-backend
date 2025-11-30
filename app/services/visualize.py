import cv2
import numpy as np
from PIL import Image

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
        final_frame = visualize_selected_segment(image, segmentation_result.masks, 
                                                 highlight_idx, boundary_color, 
                                                 thickness)
    
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

