"""Physical FOV sampling and tissue-positive tile preparation."""
import io
import cv2
import numpy as np
from PIL import Image


def patch_starts(size):
    return (np.linspace(112, size - 112, 4).round().astype(int) - 112).tolist()

def tissue_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 1] > 25) & (hsv[:, :, 2] < 245)

def slide_candidate_mask(rgb, dimensions, mpp):
    """WSI pre-sampling only; the encoder native-patch HSV rule is unchanged."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    raw = tissue_mask(rgb)
    support = (raw & (hsv[:, :, 2] > 60)).astype('uint8')
    pixel_area = dimensions[0] * mpp[0] / rgb.shape[1] * (dimensions[1] * mpp[1] / rgb.shape[0])
    umpp = np.sqrt(pixel_area)
    close_size = max(3, int(round(100 / umpp)) | 1)
    open_size = max(3, int(round(50 / umpp)) | 1)
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)))
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] * pixel_area >= 250000) + 1
    clean = np.isin(labels, keep)
    return (clean, {'candidate_mask_version': 'morphology_tissue_support', 'raw_HSV_thumbnail_fraction': float(raw.mean()), 'clean_candidate_fraction': float(clean.mean()), 'candidate_open_um': 50, 'candidate_close_um': 100, 'minimum_component_um2': 250000, 'retained_components': len(keep), 'candidate_lower_V': 60, 'candidate_filter_scope': 'WSI sampling support only; original encoder patch HSV unchanged'})

def spatial_sample(points, maximum):
    points = np.asarray(points, dtype=np.int64)
    if len(points) <= maximum:
        return points.copy()
    center = points.mean(0)
    first = np.argmin(((points - center) ** 2).sum(1))
    chosen = [first]
    distance = ((points - points[first]) ** 2).sum(1).astype(float)
    while len(chosen) < maximum:
        i = int(np.argmax(distance))
        chosen.append(i)
        distance = np.minimum(distance, ((points - points[i]) ** 2).sum(1))
        distance[chosen] = -1
    return points[np.sort(chosen)]

def candidate_fovs(dimensions, mpp, mask, fov_um, maximum):
    width, height = dimensions
    fw, fh = np.asarray(fov_um) / np.asarray(mpp)
    candidates = []
    fractions = []
    integral = cv2.integral(mask.astype('uint8'))
    for y in np.arange(0, height - fh + 1, fh):
        for x in np.arange(0, width - fw + 1, fw):
            x0, y0 = np.floor(np.array([x, y]) * [mask.shape[1] / width, mask.shape[0] / height]).astype(int)
            x1, y1 = np.ceil(np.array([x + fw, y + fh]) * [mask.shape[1] / width, mask.shape[0] / height]).astype(int)
            x1 = min(x1, mask.shape[1])
            y1 = min(y1, mask.shape[0])
            area = max(1, (x1 - x0) * (y1 - y0))
            fraction = (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / area
            if fraction >= 0.05:
                candidates.append((round(x), round(y)))
                fractions.append(fraction)
    points = np.asarray(candidates, dtype=int).reshape(-1, 2)
    selected = spatial_sample(points, maximum)
    return (selected, {'candidate_FOVs': len(points), 'selected_FOVs': len(selected), 'thumbnail_tissue_fraction': float(mask.mean()), 'thumbnail_candidate_tissue_area_um2': float(np.sum(fractions) * np.prod(fov_um)), 'FOV_native_width': float(fw), 'FOV_native_height': float(fh)})

def read_patches(slide, coords, mpp, fov_um):
    images = []
    groups = []
    fractions = []
    perbag = []
    fw, fh = np.asarray(fov_um) / np.asarray(mpp)
    for bag, (x, y) in enumerate(coords):
        region = np.asarray(slide.read_region((int(x), int(y)), 0, (round(fw), round(fh))).convert('RGB'))
        fov = cv2.resize(region, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        row = []
        for py in patch_starts(1024):
            for px in patch_starts(1024):
                patch = fov[py:py + 224, px:px + 224]
                buffer = io.BytesIO()
                Image.fromarray(patch).save(buffer, format='JPEG', quality=95, subsampling=0)
                decoded = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'))
                fraction = float(tissue_mask(decoded).mean())
                row.append(fraction)
                if fraction >= 0.05:
                    images.append(decoded)
                    groups.append(bag)
                    fractions.append(fraction)
        perbag.append(row)
    return (images, np.asarray(groups), np.asarray(perbag))
