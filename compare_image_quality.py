"""
compare_image_quality.py

Compares two images to determine which has better *actual* quality,
independent of pixel dimensions (i.e. resistant to fake upscaling).

Usage:
    python compare_image_quality.py imageA.jpg imageB.jpg

Requires:
    pip install opencv-python numpy scipy
Optional (for BRISQUE/NIQE/MUSIQ no-reference scores):
    pip install pyiqa torch
"""

import sys
import cv2
import numpy as np
from PIL import Image as PILImage

# HEIC/HEIF (the default format Apple Photos/iPhone exports) has no reliable
# OS-level decoder behind cv2.imread, so PIL needs this optional plugin
# registered before PIL.Image.open can read those files. Mirrors the
# brisque/pyiqa pattern below: a missing optional dependency must never
# crash a scan, HEIC files just fail to decode and get treated the same as
# any other unreadable file.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Maximum image dimension for FFT-based analysis. Images larger than this on
# either side are downsampled before FFT to cap memory: a 6K×4K image (192 MB
# float64) would peak at over 1.5 GB through the FFT pipeline (complex128 FFT,
# power spectrum, index arrays, radial sum); capping at 2048 keeps peak per-worker
# memory under ~35 MB without meaningful loss of frequency-cutoff accuracy (the
# metric is a fraction of Nyquist, which is scale-invariant). This constant is a
# safety bound and should not be raised without re-verifying memory usage against
# the largest images your typical scan encounters.
EFFECTIVE_RES_MAX_PX = 2048


def _load_bgr_via_pil(path):
    """Fallback decode for formats cv2 can't read at all (currently just
    HEIC/HEIF), via PIL + the registered pillow-heif opener. Returns BGR
    uint8 to match cv2's channel convention, since callers (load_gray's
    cvtColor, brisque_score) both expect BGR like every cv2.imread result.
    Returns None (rather than raising) on any decode failure -- a HEIC file
    with no HEIF plugin installed, or a genuinely corrupt file, is treated
    the same as any other unreadable file."""
    try:
        with PILImage.open(path) as pil_img:
            rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_gray(path):
    # Decodes color even though brisque (the only consumer of the BGR image)
    # is usually not installed. Decoding IMREAD_GRAYSCALE instead is ~4 ms/
    # image faster and was tried; it is NOT bit-identical, because libjpeg
    # hands back Y before chroma upsampling where BGR2GRAY weights the
    # upsampled channels. That drift is sub-1% per metric but score_group
    # min-max normalizes *within a group*, so drift that flips the sign of a
    # near-tie becomes a full 0-to-1 swing: measured over 126 synthetic
    # near-duplicate groups it moved the suggested pick in 5 of them, with no
    # evidence either arm picks better. suggested_idx is what --auto moves
    # files on. Not worth 4 ms. (tests/Test-image cannot show this -- see the
    # note in CLAUDE.md's Traps about why it is blind to scoring changes.)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        img = _load_bgr_via_pil(path)
    if img is None:
        raise FileNotFoundError(path)
    # float32, not float64: every metric below shares this one buffer, and at
    # multi-megapixel native resolution it's the dominant memory-bandwidth
    # cost in analyze() (noise_estimate/blockiness_score both do multiple
    # full-array passes over it). float32 halves that traffic; verified
    # against real photos in tests/Test-image at ~1e-7 relative drift on
    # every metric -- far below anything that could flip a quality_score
    # ranking, since float32 still carries ~7 decimal digits of precision
    # against inputs that are 8-bit pixel data to begin with.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return img, gray


def laplacian_sharpness(gray, target_size=(512, 512)):
    """Scale-normalized sharpness. Resize to a common size first,
    since Laplacian variance is not comparable across resolutions."""
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    # CV_32F, matching gray's dtype -- cv2.Laplacian doesn't support a
    # float32 source with a float64 destination depth.
    lap = cv2.Laplacian(resized, cv2.CV_32F)
    # float(): lap.var() on a float32 array returns np.float32, which --
    # unlike np.float64 -- isn't a subclass of Python float and isn't
    # JSON-serializable, breaking find_duplicates.py's on-disk analyze cache.
    return float(lap.var())


def _power_spectrum(gray):
    """Radially-symmetric power spectrum |FFT|^2, via whichever backend is
    actually fast for these dimensions.

    cv2.dft is ~1.7x faster than np.fft.fft2 and, like every other cv2 call
    in this codebase, properly releases the GIL (numpy's FFT only partially
    does, capping thread-pool scaling) -- but only when both dimensions are
    already DFT-efficient. OpenCV's DFT degrades badly otherwise: measured
    up to ~24x *slower* than np.fft (3.2s vs 0.13s on a 2039x2039 array,
    2039 being prime) on sizes with large prime factors, since np.fft.fft2
    always runs Bluestein's algorithm (guaranteed O(n log n)) while cv2.dft
    does not. cv2.getOptimalDFTSize(n) == n exactly when n is 5-smooth
    (only 2/3/5 as prime factors) -- the sizes cv2.dft handles efficiently --
    so that equality is used as the routing check rather than guessing from
    the raw dimension. effective_resolution()'s downsample-to-2048 step
    produces arbitrary post-scale dimensions on the non-capped axis, and
    un-downsampled images carry whatever native dimensions the source photo
    has, so this check is live in real usage, not just a defensive-only
    branch -- verified via tests/Test-image's real photos (none of which
    happen to hit the slow case) plus direct synthetic timing at prime
    sizes."""
    if cv2.getOptimalDFTSize(gray.shape[0]) == gray.shape[0] and cv2.getOptimalDFTSize(gray.shape[1]) == gray.shape[1]:
        dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=(0, 1))
        return dft_shift[..., 0].astype(np.float64) ** 2 + dft_shift[..., 1].astype(np.float64) ** 2
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    return np.abs(fshift) ** 2


def effective_resolution(gray):
    """
    Estimate the true information cutoff frequency via the radially
    averaged power spectrum. Returns:
      - cutoff_fraction: fraction of Nyquist frequency where real signal
        ends (1.0 = uses full native resolution, lower = effectively
        upscaled/soft beyond that point)
      - equivalent_pixels: cutoff_fraction * min(h, w), a rough proxy for
        "true" linear resolution regardless of stored dimensions
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0, 0.0  # too small for any meaningful frequency analysis

    # Downsample large images before FFT to cap memory. The cutoff_fraction
    # is a fraction of Nyquist (scale-invariant), so this doesn't change the
    # result. equivalent_pixels below uses the ORIGINAL min(h, w) so the
    # absolute estimate remains correct regardless of downsampling.
    orig_min_side = min(h, w)
    if max(h, w) > EFFECTIVE_RES_MAX_PX:
        scale = EFFECTIVE_RES_MAX_PX / max(h, w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.0, 0.0  # extreme aspect ratio: too thin after the cap

    power = _power_spectrum(gray)

    cy, cx = h // 2, w // 2
    # Broadcast two 1-D squared-offset arrays instead of np.indices' two full
    # h*w int64 grids: same int64 arithmetic and same float64 sqrt, so the
    # result is bit-identical, at ~1.7x the speed and one fewer big temporary.
    dx2 = (np.arange(w) - cx) ** 2
    dy2 = (np.arange(h) - cy) ** 2
    r = np.sqrt(dx2[None, :] + dy2[:, None]).astype(np.int32)
    max_r = min(cx, cy)

    radial_power = np.bincount(r.ravel(), power.ravel())[:max_r]
    radial_power = radial_power / (np.arange(1, max_r + 1))  # normalize by ring area
    log_power = np.log(radial_power + 1e-8)

    # Smooth and find where the spectrum flattens into the noise floor.
    kernel = np.ones(5) / 5
    smoothed = np.convolve(log_power, kernel, mode="same")

    noise_floor = np.median(smoothed[-max(5, max_r // 20):])
    threshold = noise_floor + 0.5  # tolerance above floor

    cutoff_idx = 1  # fallback: near-zero when no ring exceeds the floor
    for i in range(max_r - 1, 0, -1):
        if smoothed[i] > threshold:
            cutoff_idx = i
            break

    cutoff_fraction = cutoff_idx / max_r
    equivalent_pixels = cutoff_fraction * orig_min_side
    return cutoff_fraction, equivalent_pixels


def noise_estimate(gray):
    """Fast noise sigma estimate (Immerkaer's method)."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0  # noise indistinguishable from signal at this size
    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    # float32, matching gray's dtype -- cv2.filter2D with ddepth=-1 (same
    # depth as source) needs the kernel's dtype to match the source's.
    M = np.array(M, dtype=np.float32)
    conv = cv2.filter2D(gray, -1, M)
    # cv2.norm(..., NORM_L1), not np.sum(np.abs(conv)): fuses the abs and the
    # reduction into one C call instead of two full-array numpy passes (one
    # to build the abs() temporary, one to sum it) -- ~2.7x faster, verified
    # against real photos in tests/Test-image at 0 relative drift.
    l1 = cv2.norm(conv, cv2.NORM_L1)
    sigma = l1 * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    return sigma


def blockiness_score(gray, block_size=8):
    """Detects JPEG-style blocking artifacts by measuring discontinuity
    strength at block boundaries vs within blocks."""
    h, w = gray.shape
    # cv2.absdiff, not np.abs(np.diff(...)): fuses the subtract and the abs
    # into one C call instead of two full-array numpy passes -- ~2.3-2.5x
    # faster, verified against real photos in tests/Test-image at ~1e-7
    # relative drift.
    diff_h = cv2.absdiff(gray[:, 1:], gray[:, :-1])
    diff_v = cv2.absdiff(gray[1:, :], gray[:-1, :])

    boundary_cols = np.arange(block_size - 1, w - 1, block_size)
    boundary_rows = np.arange(block_size - 1, h - 1, block_size)

    boundary_energy_h = diff_h[:, boundary_cols].mean() if len(boundary_cols) else 0
    boundary_energy_v = diff_v[boundary_rows, :].mean() if len(boundary_rows) else 0
    overall_h = diff_h.mean()
    overall_v = diff_v.mean()

    score = ((boundary_energy_h - overall_h) + (boundary_energy_v - overall_v)) / 2
    # float(): the .mean() calls above run on float32 arrays now, so score is
    # np.float32 -- unlike np.float64, not JSON-serializable (see
    # laplacian_sharpness for the same fix and why it matters).
    return float(max(score, 0))


def brisque_score(img_bgr):
    """
    BRISQUE via the `brisque` package (pip install brisque[opencv-python-headless]).
    Correct API is class-based (BRISQUE().score()), not a module-level function.
    Requires RGB input, not BGR, since the package's reference implementation
    builds its ndarray from PIL (RGB) images.
    Lower score = better perceived quality.
    """
    try:
        from brisque import BRISQUE
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        obj = BRISQUE(url=False)
        return float(obj.score(img=img_rgb))
    except Exception:
        return None


def niqe_score(path):
    """Optional secondary check via pyiqa (pip install pyiqa torch). Lower = better."""
    try:
        import pyiqa
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        niqe = pyiqa.create_metric("niqe", device=device)
        return float(niqe(path))
    except Exception:
        return None


def analyze(path):
    img, gray = load_gray(path)
    sharpness = laplacian_sharpness(gray)
    cutoff_fraction, eq_px = effective_resolution(gray)
    noise = noise_estimate(gray)
    blockiness = blockiness_score(gray)
    brisque = brisque_score(img)
    niqe = niqe_score(path)

    return {
        "path": path,
        "dimensions": (gray.shape[1], gray.shape[0]),
        "sharpness_normalized": sharpness,
        "effective_resolution_fraction": cutoff_fraction,
        "effective_resolution_px_equiv": eq_px,
        "noise_sigma": noise,
        "blockiness": blockiness,
        "brisque": brisque,
        "niqe": niqe,
    }


def report(a, b):
    print(f"\n{'Metric':30s} {'Image A':>18s} {'Image B':>18s}")
    print("-" * 68)
    print(f"{'Dimensions':30s} {str(a['dimensions']):>18s} {str(b['dimensions']):>18s}")
    print(f"{'Sharpness (normalized)':30s} {a['sharpness_normalized']:18.2f} {b['sharpness_normalized']:18.2f}")
    print(f"{'Effective res. fraction':30s} {a['effective_resolution_fraction']:18.3f} {b['effective_resolution_fraction']:18.3f}")
    print(f"{'Effective res. (px equiv)':30s} {a['effective_resolution_px_equiv']:18.1f} {b['effective_resolution_px_equiv']:18.1f}")
    print(f"{'Noise sigma (lower=cleaner)':30s} {a['noise_sigma']:18.3f} {b['noise_sigma']:18.3f}")
    print(f"{'Blockiness (lower=better)':30s} {a['blockiness']:18.3f} {b['blockiness']:18.3f}")
    if a["brisque"] is not None and b["brisque"] is not None:
        print(f"{'BRISQUE (lower=better)':30s} {a['brisque']:18.2f} {b['brisque']:18.2f}")
    else:
        print("\n(Install brisque[opencv-python-headless] for BRISQUE)")

    if a["niqe"] is not None and b["niqe"] is not None:
        print(f"{'NIQE (lower=better)':30s} {a['niqe']:18.2f} {b['niqe']:18.2f}")
    else:
        print("(Install pyiqa + torch for NIQE)")

    print("\nInterpretation:")
    print("- Higher 'effective resolution fraction' = less upscaled / more real detail per pixel.")
    print("- Higher sharpness (at matched scale) = more true detail, not just size.")
    print("- Lower noise, lower blockiness, lower BRISQUE/NIQE = better perceived quality.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_image_quality.py imageA imageB")
        sys.exit(1)

    result_a = analyze(sys.argv[1])
    result_b = analyze(sys.argv[2])
    report(result_a, result_b)
