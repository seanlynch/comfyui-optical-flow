import numpy as np
import cv2
import torch


def np2tensor(img_np: np.ndarray | list[np.ndarray]) -> torch.Tensor:
    if isinstance(img_np, list):
        return torch.cat([np2tensor(img) for img in img_np], dim=0)

    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)


def tensor2np(tensor: torch.Tensor) -> list[np.ndarray]:
    batch_count = tensor.size(0) if len(tensor.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2np(tensor[i]))
        return out

    return [np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)]


def remap(img, flow, border_mode = cv2.BORDER_REFLECT_101):
    # copyMakeBorder doesn't support wrap, but supports replicate. Replaces wrap with reflect101.
    if border_mode == cv2.BORDER_WRAP:
        border_mode = cv2.BORDER_REFLECT_101
    h, w = img.shape[:2]
    displacement = int(h * 0.25), int(w * 0.25)
    larger_img = cv2.copyMakeBorder(img, displacement[0], displacement[0], displacement[1], displacement[1], border_mode)
    lh, lw = larger_img.shape[:2]
    larger_flow = extend_flow(flow, lw, lh)
    remapped_img = cv2.remap(larger_img, larger_flow, None, cv2.INTER_LINEAR, border_mode)
    output_img = center_crop_image(remapped_img, w, h)
    return output_img


def center_crop_image(img, w, h):
    y, x, _ = img.shape
    width_indent = int((x - w) / 2)
    height_indent = int((y - h) / 2)
    cropped_img = img[height_indent:y-height_indent, width_indent:x-width_indent]
    return cropped_img


def extend_flow(flow, w, h):
    # Get the shape of the original flow image
    flow_h, flow_w = flow.shape[:2]
    # Calculate the position of the image in the new image
    x_offset = int((w - flow_w) / 2)
    y_offset = int((h - flow_h) / 2)
    # Generate the X and Y grids
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    # Create the new flow image and set it to the X and Y grids
    new_flow = np.dstack((x_grid, y_grid)).astype(np.float32)
    # Shift the values of the original flow by the size of the border
    flow[:,:,0] += x_offset
    flow[:,:,1] += y_offset
    # Overwrite the middle of the grid with the original flow
    new_flow[y_offset:y_offset+flow_h, x_offset:x_offset+flow_w, :] = flow
    # Return the extended image
    return new_flow


def get_flow_from_images(i1, i2, method, prev_flow=None):
    if method == "DIS Medium":
        flow = get_flow_from_images_DIS(i1, i2, 'medium', prev_flow)
    elif method == "DIS Fine":
        flow = get_flow_from_images_DIS(i1, i2, 'fine', prev_flow)
    elif method == "Farneback": # Farneback Normal:
        flow = get_flow_from_images_Farneback(i1, i2, prev_flow)
    else:
        # if we reached this point, something went wrong. raise an error:
        raise RuntimeError(f"Invald flow method name: '{method}'")

    return flow


def get_flow_from_images_DIS(i1, i2, preset, prev_flow):
    # DIS PRESETS CHART KEY: finest scale, grad desc its, patch size
    # DIS_MEDIUM: 1, 25, 8 | DIS_FAST: 2, 16, 8 | DIS_ULTRAFAST: 2, 12, 8
    if preset == 'medium': preset_code = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM    
    elif preset == 'fast': preset_code = cv2.DISOPTICAL_FLOW_PRESET_FAST    
    elif preset == 'ultrafast': preset_code = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST   
    elif preset in ['slow','fine']: preset_code = None
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(preset_code)
    # custom presets
    if preset == 'slow':
        dis.setGradientDescentIterations(192)
        dis.setFinestScale(1)
        dis.setPatchSize(8)
        dis.setPatchStride(4)
    if preset == 'fine':
        dis.setGradientDescentIterations(192)
        dis.setFinestScale(0)
        dis.setPatchSize(8)
        dis.setPatchStride(4)
    return dis.calc(i1, i2, prev_flow)


def get_flow_from_images_Farneback(i1, i2, preset="normal", last_flow=None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0):
    flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN         # Specify the operation flags
    pyr_scale = 0.5   # The image scale (<1) to build pyramids for each image
    if preset == "fine":
        levels = 13       # The number of pyramid layers, including the initial image
        winsize = 77      # The averaging window size
        iterations = 13   # The number of iterations at each pyramid level
        poly_n = 15       # The size of the pixel neighborhood used to find polynomial expansion in each pixel
        poly_sigma = 0.8  # The standard deviation of the Gaussian used to smooth derivatives used as a basis for the polynomial expansion
    else: # "normal"
        levels = 5        # The number of pyramid layers, including the initial image
        winsize = 21      # The averaging window size
        iterations = 5    # The number of iterations at each pyramid level
        poly_n = 7        # The size of the pixel neighborhood used to find polynomial expansion in each pixel
        poly_sigma = 1.2  # The standard deviation of the Gaussian used to smooth derivatives used as a basis for the polynomial expansion
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    flags = 0 # flags = cv2.OPTFLOW_USE_INITIAL_FLOW    
    flow = cv2.calcOpticalFlowFarneback(i1, i2, last_flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    return flow


def image_transform_optical_flow(img, flow, border_mode=cv2.BORDER_REPLICATE, flow_reverse=False):
    if not flow_reverse:
        flow = -flow
    h, w = img.shape[:2]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:,np.newaxis]
    return remap(img, flow, border_mode)


def draw_flow_lines_in_grid_in_color(img, flow, step=8, magnitude_multiplier=1, min_magnitude = 0, max_magnitude = 10000):
    flow = flow * magnitude_multiplier
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    vis = cv2.add(vis, bgr)

    # Iterate through the lines
    for (x1, y1), (x2, y2) in lines:
        # Calculate the magnitude of the line
        magnitude = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Only draw the line if it falls within the magnitude range
        if min_magnitude <= magnitude <= max_magnitude:
            b = int(bgr[y1, x1, 0])
            g = int(bgr[y1, x1, 1])
            r = int(bgr[y1, x1, 2])
            color = (b, g, r)
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), color, thickness=1, tipLength=0.1)    
    return vis


def visualize_flow(flow_img, flow):
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2GRAY)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_GRAY2BGR)
    flow_img = draw_flow_lines_in_grid_in_color(flow_img, flow)
    return cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)


class ComputeOpticalFlow:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prev": ("IMAGE",),
                "current": ("IMAGE",),
                "method": ([
                    "DIS Medium",
                    "DIS Fine",
                    "Farneback",
                ],),
            },
        }

    RETURN_TYPES = ("OPTICAL_FLOW",)
    FUNCTION = "compute_flow"
    CATEGORY = "Optical flow"

    def compute_flow(self, prev, current, method):
        images = zip(tensor2np(prev), tensor2np(current))
        return ([get_flow_from_images(im1, im2, method) for im1, im2 in images],)


class ApplyOpticalFlow:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "flow": ("OPTICAL_FLOW",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_flow"
    CATEGORY = "Optical flow"

    def apply_flow(self, image, flow):
        ims = tensor2np(image)
        out = [image_transform_optical_flow(im, f) for im, f in zip(ims, flow)]
        return (np2tensor(out),)


class VisualizeOpticalFlow:
    """Visualize a flow as a set of arrows superimposed on the original image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "flow": ("OPTICAL_FLOW",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_flow"
    CATEGORY = "Optical flow"

    def visualize_flow(self, image, flow):
        ifs = zip(tensor2np(image), flow)
        out = [visualize_flow(img, flow) for img, flow in ifs]
        return (np2tensor(out),)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Compute optical flow": ComputeOpticalFlow,
    "Apply optical flow": ApplyOpticalFlow,
    "Visualize optical flow": VisualizeOpticalFlow,
}


# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComputeOpticalFlow": "Compute optical flow",
    "ApplyOpticalFlow": "Apply optical flow",
    "VisualizeOpticalFlow": "Visualize optical flow",
}
