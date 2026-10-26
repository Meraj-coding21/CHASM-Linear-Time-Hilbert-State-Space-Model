"""SRAD + dual-tree-style wavelet preprocessing used in the CHASM notebook.

The dual-tree implementation follows the released notebook: two db4 DWT trees
are formed using a half-sample shifted copy, thresholded as complex pairs, then
averaged after inverse reconstruction. This module removes notebook-only plotting
and metric code but preserves the processing path.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import pywt
from scipy import ndimage
from skimage import color, exposure, transform


class VectorizedSRAD:
    def __init__(self, num_iterations: int = 15, delta_t: float = 0.05, kappa: float = 30.0):
        self.num_iterations = int(num_iterations)
        self.delta_t = float(delta_t)
        self.kappa = float(kappa)
        self.eps = 1e-8

    def compute_gradients(self, image):
        padded = np.pad(image, 1, mode="edge")
        return (
            padded[:-2, 1:-1] - image,
            padded[2:, 1:-1] - image,
            padded[1:-1, 2:] - image,
            padded[1:-1, :-2] - image,
        )

    def compute_diffusion_coefficient(self, gradient, q0_squared):
        laplacian = gradient / (self.kappa ** 2)
        numerator = laplacian - q0_squared
        denominator = q0_squared * (1 + q0_squared) + self.eps
        coefficient = 1.0 / (1.0 + numerator / denominator)
        return np.clip(coefficient, 0, 1)

    def estimate_speckle_parameter(self, image):
        h, w = image.shape
        roi = image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        mean_val = np.mean(roi) + self.eps
        return (np.std(roi) / mean_val) ** 2

    def diffuse(self, image):
        image = np.asarray(image, dtype=np.float64)
        img_min, img_max = float(image.min()), float(image.max())
        if img_max - img_min <= self.eps:
            return image.copy()
        img = (image - img_min) / (img_max - img_min + self.eps)
        img = img + self.eps
        q0_squared = self.estimate_speckle_parameter(img)
        for _ in range(self.num_iterations):
            gradients = self.compute_gradients(img)
            coeffs = [
                self.compute_diffusion_coefficient((gradient / img) ** 2, q0_squared)
                for gradient in gradients
            ]
            divergence = sum(c * g for c, g in zip(coeffs, gradients))
            img = np.clip(img + self.delta_t * divergence, self.eps, 1.0)
        return img * (img_max - img_min) + img_min


class DualTreeCWT:
    def __init__(self, wavelet: str = "db4", levels: int = 2):
        self.wavelet = wavelet
        self.levels = int(levels)
        self.shift_grids = {}

    def _prepare_shift_grids(self, shape):
        if shape in self.shift_grids:
            return self.shift_grids[shape]
        h, w = shape
        y = np.clip(np.arange(h) + 0.5, 0, h - 1)
        x = np.clip(np.arange(w) + 0.5, 0, w - 1)
        coords = np.meshgrid(x, y, indexing="xy")
        self.shift_grids[shape] = np.array([coords[1], coords[0]])
        return self.shift_grids[shape]

    def _half_sample_delay(self, image):
        return ndimage.map_coordinates(image, self._prepare_shift_grids(image.shape), order=1, mode="reflect")

    @staticmethod
    def _inverse_half_sample_delay(image):
        h, w = image.shape
        y = np.clip(np.arange(h) - 0.5, 0, h - 1)
        x = np.clip(np.arange(w) - 0.5, 0, w - 1)
        coords = np.meshgrid(x, y, indexing="xy")
        return ndimage.map_coordinates(image, np.array([coords[1], coords[0]]), order=1, mode="reflect")

    def decompose(self, image):
        tree1 = pywt.wavedec2(image, self.wavelet, level=self.levels)
        tree2 = pywt.wavedec2(self._half_sample_delay(image), self.wavelet, level=self.levels)
        complex_coeffs = [tree1[0]]
        for coeff1, coeff2 in zip(tree1[1:], tree2[1:]):
            complex_coeffs.append(tuple(a + 1j * b for a, b in zip(coeff1, coeff2)))
        return complex_coeffs

    def reconstruct(self, complex_coeffs):
        tree1, tree2 = [complex_coeffs[0]], [complex_coeffs[0]]
        for coeff in complex_coeffs[1:]:
            tree1.append(tuple(np.real(x) for x in coeff))
            tree2.append(tuple(np.imag(x) for x in coeff))
        recon1 = pywt.waverec2(tree1, self.wavelet)
        recon2 = pywt.waverec2(tree2, self.wavelet)
        return (recon1 + self._inverse_half_sample_delay(recon2)) / 2.0


class HybridGallbladderDenoiser:
    def __init__(self, srad_iterations: int = 15, wavelet: str = "db4", dwt_levels: int = 2):
        self.srad = VectorizedSRAD(num_iterations=srad_iterations)
        self.dtcwt = DualTreeCWT(wavelet=wavelet, levels=dwt_levels)
        self.eps = 1e-8

    def normalize_image(self, image):
        img_min, img_max = float(image.min()), float(image.max())
        if img_max - img_min > self.eps:
            return (image - img_min) / (img_max - img_min)
        return image.copy()

    def extract_lumen_patch(self, image, patch_size=(40, 40)):
        h, w = image.shape
        ph = min(patch_size[0], max(4, h // 3))
        pw = min(patch_size[1], max(4, w // 3))
        y0, y1 = int(h * 0.30), int(h * 0.70)
        x0, x1 = int(w * 0.20), int(w * 0.80)
        best_patch, min_mean = None, np.inf
        for y in range(y0, max(y0 + 1, y1 - ph + 1), 10):
            for x in range(x0, max(x0 + 1, x1 - pw + 1), 10):
                patch = image[y : y + ph, x : x + pw]
                if patch.size == 0:
                    continue
                mean, variance = float(np.mean(patch)), float(np.var(patch))
                if mean < 0.01 or variance > 0.05:
                    continue
                if mean < min_mean:
                    best_patch, min_mean = patch.copy(), mean
        if best_patch is None:
            cy, cx = h // 2, w // 2
            ys = max(0, min(h - ph, cy - ph // 2))
            xs = max(0, min(w - pw, cx - pw // 2))
            best_patch = image[ys : ys + ph, xs : xs + pw].copy()
        return best_patch

    def estimate_noise_variance_complex(self, coeffs):
        output = {"H": [], "V": [], "D": []}
        for c_h, c_v, c_d in coeffs[1:]:
            output["H"].append(float(np.var(np.abs(c_h))) + self.eps)
            output["V"].append(float(np.var(np.abs(c_v))) + self.eps)
            output["D"].append(float(np.var(np.abs(c_d))) + self.eps)
        return output

    def bayes_shrink_threshold(self, magnitude, noise_var):
        signal_var = float(np.var(magnitude))
        sigma_s = np.sqrt(max(0.0, signal_var - noise_var) + self.eps)
        threshold = noise_var / (sigma_s + self.eps)
        return min(float(threshold), float(np.median(magnitude)))

    @staticmethod
    def complex_soft_threshold(coeff, threshold):
        magnitude, phase = np.abs(coeff), np.angle(coeff)
        return np.maximum(magnitude - threshold, 0) * np.exp(1j * phase)

    def apply_clahe_to_ll(self, ll, clip_limit=0.02, kernel_size=(16, 16)):
        ll_norm = self.normalize_image(ll)
        enhanced = exposure.equalize_adapthist(ll_norm, clip_limit=clip_limit, kernel_size=kernel_size)
        ll_min, ll_max = float(ll.min()), float(ll.max())
        return enhanced * (ll_max - ll_min) + ll_min

    def denoise(self, image):
        image = np.asarray(image, dtype=np.float64)
        original_shape = image.shape
        small_shape = (max(32, image.shape[0] // 2), max(32, image.shape[1] // 2))
        image_small = transform.resize(image, small_shape, anti_aliasing=True, preserve_range=True)
        srad_output = self.srad.diffuse(image_small)
        normalized = self.normalize_image(srad_output)
        coeffs = self.dtcwt.decompose(normalized)
        noise_coeffs = self.dtcwt.decompose(self.extract_lumen_patch(normalized))
        noise = self.estimate_noise_variance_complex(noise_coeffs)
        denoised = [coeffs[0]]
        for level, (c_h, c_v, c_d) in enumerate(coeffs[1:]):
            i = min(level, len(noise["H"]) - 1)
            denoised.append((
                self.complex_soft_threshold(c_h, self.bayes_shrink_threshold(np.abs(c_h), noise["H"][i])),
                self.complex_soft_threshold(c_v, self.bayes_shrink_threshold(np.abs(c_v), noise["V"][i])),
                self.complex_soft_threshold(c_d, self.bayes_shrink_threshold(np.abs(c_d), noise["D"][i])),
            ))
        denoised[0] = self.apply_clahe_to_ll(denoised[0])
        output = self.dtcwt.reconstruct(denoised)
        output = transform.resize(output, original_shape, anti_aliasing=True, preserve_range=True)
        return srad_output, np.clip(output, 0.0, max(float(image.max()), 1e-8))


def load_grayscale(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    return arr


def save_grayscale(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(image), 0.0, 1.0)
    Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L").save(path)
