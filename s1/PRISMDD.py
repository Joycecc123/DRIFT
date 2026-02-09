import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.io import imread
from matplotlib.patches import Arc, Circle


class DatasetRatioCalculator:
    """
    Estimate mixing ratios between Dataset A and Dataset B using:
      1) Category distribution bias (signed + magnitude)
      2) Pixel intensity mean difference (signed + magnitude)
      3) Pixel intensity std difference (signed + magnitude)
      4) GLCM contrast difference (signed + magnitude)

    Key improvements vs. older versions:
      - Category bias is aggregated into scalar features (no arrays leaking into ratio calc).
      - Shift has DIRECTION (can favor A or B) using signed differences.
      - Shift magnitude is controlled by total difference (magnitude gate) and direction (sign gate).
      - Confidence is computed from feature significance + consistency + intensity.
    """

    def __init__(self):
        # Feature weights (how much each feature contributes)
        self.feature_weights = {
            "category_distribution": 0.40,
            "pixel_mean": 0.20,
            "pixel_std": 0.20,
            "glcm_contrast": 0.20
        }

        # Per-class weights used when computing category distribution (you can tune these)
        self.class_weights = {
            0: 0.10,  # background
            1: 0.60,  # tumor
            2: 0.30   # stroma
        }

        # Scales used to normalize raw feature values
        self.feature_scales = {
            "category_distribution": 1.0,    # already in [0, 1] scale after aggregation
            "pixel_mean": 255.0,             # grayscale range
            "pixel_std": 255.0,
            "glcm_contrast": 2000.0
        }

        # Ratio control
        self.max_shift = 0.15               # max deviation from 0.5 (=> [0.35, 0.65])
        self.shift_sensitivity = 3.0        # sigmoid sensitivity for total difference
        self.direction_sensitivity = 2.5    # tanh sensitivity for direction score

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _safe_tanh(x: float) -> float:
        return float(np.tanh(x))

    def normalize_features(self, bias_details: dict) -> dict:
        """
        bias_details expects:
          - "category_bias": dict with keys:
                "weighted_abs_bias_sum": float >= 0
                "weighted_signed_bias_sum": float (can be +/-)
          - "pixel_mean_diff": float (signed, mean_a - mean_b)
          - "pixel_std_diff": float (signed, std_a - std_b)
          - "glcm_contrast_diff": float (signed, glcm_a - glcm_b)
        """
        # Magnitude features (non-negative)
        cat_mag = bias_details["category_bias"]["weighted_abs_bias_sum"] / self.feature_scales["category_distribution"]
        mean_mag = abs(bias_details["pixel_mean_diff"]) / self.feature_scales["pixel_mean"]
        std_mag = abs(bias_details["pixel_std_diff"]) / self.feature_scales["pixel_std"]
        glcm_mag = abs(bias_details["glcm_contrast_diff"]) / self.feature_scales["glcm_contrast"]

        # Signed direction features (can be +/-)
        cat_dir = bias_details["category_bias"]["weighted_signed_bias_sum"] / self.feature_scales["category_distribution"]
        mean_dir = bias_details["pixel_mean_diff"] / self.feature_scales["pixel_mean"]
        std_dir = bias_details["pixel_std_diff"] / self.feature_scales["pixel_std"]
        glcm_dir = bias_details["glcm_contrast_diff"] / self.feature_scales["glcm_contrast"]

        # Non-linear squashing to reduce sensitivity to tiny differences
        normalized = {
            "magnitude": {
                "category_distribution": self._safe_tanh(cat_mag),
                "pixel_mean": self._safe_tanh(mean_mag),
                "pixel_std": self._safe_tanh(std_mag),
                "glcm_contrast": self._safe_tanh(glcm_mag),
            },
            "direction": {
                "category_distribution": self._safe_tanh(cat_dir),
                "pixel_mean": self._safe_tanh(mean_dir),
                "pixel_std": self._safe_tanh(std_dir),
                "glcm_contrast": self._safe_tanh(glcm_dir),
            }
        }
        return normalized

    def calculate_feature_importance(self, normalized_magnitude: dict) -> dict:
        """
        Convert normalized magnitude into importance scores.
        Importance is always >= 0 and multiplied by feature_weights.
        """
        importance_scores = {}
        for feature, value in normalized_magnitude.items():
            # A mild sigmoid mapping; value in [0, 1)
            imp = self._sigmoid(3.0 * (value - 0.2))
            importance_scores[feature] = imp * self.feature_weights[feature]
        return importance_scores

    def calculate_confidence(self, normalized_magnitude: dict, importance_scores: dict) -> float:
        """
        Confidence in [0, 1], based on:
          - weighted significance of features
          - consistency across features
          - difference intensity (importance-weighted magnitudes)
        """
        # Significance from normalized magnitudes
        significance = {f: self._sigmoid(5.0 * (v - 0.15)) for f, v in normalized_magnitude.items()}

        weighted_significance = sum(significance[f] * self.feature_weights[f] for f in normalized_magnitude)
        total_weight = sum(self.feature_weights.values())

        vals = np.array(list(significance.values()), dtype=float)
        consistency = 1.0 - (np.std(vals) / (np.mean(vals) + 1e-6))

        diff_intensity = float(np.mean([normalized_magnitude[f] * importance_scores[f] * 0.8 for f in normalized_magnitude]))

        base_conf = 0.4 * (weighted_significance / total_weight) + 0.4 * consistency + 0.2 * diff_intensity
        enhanced = self._sigmoid(4.0 * (base_conf - 0.3))

        if consistency > 0.7:
            enhanced *= 1.2

        return float(np.clip(enhanced, 0.0, 1.0))

    def calculate_dataset_ratios(self, bias_details: dict) -> dict:
        normalized = self.normalize_features(bias_details)
        mag = normalized["magnitude"]
        direc = normalized["direction"]

        importance = self.calculate_feature_importance(mag)

        # Total difference (magnitude gate): bigger => larger shift magnitude
        total_difference = float(sum(importance.values()))
        magnitude_gate = (2.0 / (1.0 + np.exp(-self.shift_sensitivity * total_difference)) - 1.0)  # in (0, 1)

        # Direction score (sign gate): can be positive or negative
        # Weighted by importance so more important features dominate direction
        direction_raw = float(sum(direc[f] * (importance[f] + 1e-6) for f in importance))
        direction_gate = float(np.tanh(self.direction_sensitivity * direction_raw))  # in (-1, 1)

        # Final shift: can be +/- and bounded
        shift = self.max_shift * magnitude_gate * direction_gate

        ratio_a = float(np.clip(0.5 + shift, 0.5 - self.max_shift, 0.5 + self.max_shift))
        ratio_b = 1.0 - ratio_a

        confidence = self.calculate_confidence(mag, importance)

        return {
            "ratio_a": ratio_a,
            "ratio_b": ratio_b,
            "shift": shift,
            "direction_gate": direction_gate,
            "magnitude_gate": magnitude_gate,
            "total_difference": total_difference,
            "normalized_magnitude": mag,
            "normalized_direction": direc,
            "importance_scores": importance,
            "confidence": confidence
        }


# ---------------- Utility Functions ----------------

def load_images_and_masks(images_path: str, masks_path: str):
    images, masks = [], []
    for f in sorted(os.listdir(images_path)):
        if f.lower().endswith(".png"):
            img = imread(os.path.join(images_path, f))
            mask = imread(os.path.join(masks_path, f))

            if img.ndim == 3:
                img = rgb2gray(img) * 255.0

            images.append(img.astype("uint8"))
            masks.append(mask.astype("uint8"))
    return images, masks


def crop_images_and_masks(images_a, masks_a, images_b, masks_b):
    min_h = min(
        min(img.shape[0] for img in images_a + images_b),
        min(msk.shape[0] for msk in masks_a + masks_b)
    )
    min_w = min(
        min(img.shape[1] for img in images_a + images_b),
        min(msk.shape[1] for msk in masks_a + masks_b)
    )
    print(f"Cropping to common size: {min_h}x{min_w}")

    crop = lambda lst: [x[:min_h, :min_w] for x in lst]
    return crop(images_a), crop(masks_a), crop(images_b), crop(masks_b)


def calculate_category_bias(masks_a, masks_b, class_weights: dict):
    """
    Returns:
      bias_info: dict with
        - weighted_abs_bias_sum: sum_w |pA(c)-pB(c)|
        - weighted_signed_bias_sum: sum_w (pA(c)-pB(c))   (directional)
      category_ratios: dict with per-class ratios for A and B
    """
    flat_a = np.concatenate([m.flatten() for m in masks_a])
    flat_b = np.concatenate([m.flatten() for m in masks_b])

    categories = np.unique(np.concatenate([flat_a, flat_b]))

    category_ratios = {"A": {}, "B": {}}
    weighted_abs = []
    weighted_signed = []

    for c in categories:
        p_a = float(np.sum(flat_a == c) / flat_a.size)
        p_b = float(np.sum(flat_b == c) / flat_b.size)

        category_ratios["A"][int(c)] = p_a
        category_ratios["B"][int(c)] = p_b

        w = float(class_weights.get(int(c), 1.0))
        diff = p_a - p_b

        weighted_abs.append(abs(diff) * w)
        weighted_signed.append(diff * w)

        print(f"Class {int(c)}: pA={p_a:.4f}, pB={p_b:.4f}, w={w:.3f}, signed={diff*w:+.4f}, abs={abs(diff)*w:.4f}")

    bias_info = {
        "weighted_abs_bias_sum": float(np.sum(weighted_abs)),
        "weighted_signed_bias_sum": float(np.sum(weighted_signed))
    }
    return bias_info, category_ratios


def calculate_pixel_stats(images_a, images_b):
    mean_a = float(np.mean([np.mean(i) for i in images_a]))
    mean_b = float(np.mean([np.mean(i) for i in images_b]))
    std_a = float(np.mean([np.std(i) for i in images_a]))
    std_b = float(np.mean([np.std(i) for i in images_b]))
    return mean_a, mean_b, std_a, std_b


def calculate_glcm_contrast(images):
    contrasts = []
    for img in images:
        glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        contrasts.append(float(contrast))
    return float(np.mean(contrasts))


# ---------------- Plotting ----------------

def plot_feature_importance(importance_scores: dict, normalized_magnitude: dict, save_path="feature_importance.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    features = list(importance_scores.keys())
    scores = [importance_scores[f] for f in features]
    mags = [normalized_magnitude[f] for f in features]
    colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]

    # Left: importance
    bars1 = ax1.barh(features, scores, color=colors)
    ax1.set_title("Feature Importance Scores", pad=20)
    ax1.set_xlabel("Importance Score")
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", ha="left", va="center", fontsize=10)

    # Right: normalized magnitudes
    bars2 = ax2.barh(features, mags, color=colors)
    ax2.set_title("Normalized Feature Magnitudes", pad=20)
    ax2.set_xlabel("Normalized Magnitude (tanh-scaled)")
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", ha="left", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_combined_analysis(results: dict, save_path="combined_analysis.png"):
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 3, figure=fig)

    # Pie chart for ratio
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie(
        [results["ratio_a"], results["ratio_b"]],
        labels=[f"Dataset A\n({results['ratio_a']:.3f})", f"Dataset B\n({results['ratio_b']:.3f})"],
        colors=["#FF9999", "#66B2FF"],
        autopct="%1.1f%%"
    )
    ax1.set_title("Dataset Mixing Ratio", pad=20)

    # Importance bars
    ax2 = fig.add_subplot(gs[0, 1:])
    features = list(results["importance_scores"].keys())
    imps = [results["importance_scores"][f] for f in features]
    ax2.barh(features, imps, color=["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"])
    ax2.set_title("Feature Importance Scores", pad=20)
    for i, v in enumerate(imps):
        ax2.text(v, i, f"{v:.3f}", va="center", ha="left")

    # Confidence gauge
    ax3 = fig.add_subplot(gs[1, 0])
    conf = results["confidence"]
    ax3.add_patch(Circle((0.5, 0.5), 0.4, color="lightgray"))
    ax3.add_patch(Arc((0.5, 0.5), 0.8, 0.8, theta1=0, theta2=360 * conf, color="#FF9999", linewidth=10))
    ax3.text(0.5, 0.3, f"Confidence\n{conf:.3f}", ha="center", va="center", fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    # Normalized magnitudes
    ax4 = fig.add_subplot(gs[1, 1:])
    mags = [results["normalized_magnitude"][f] for f in features]
    ax4.barh(features, mags, color=["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"])
    ax4.set_title("Normalized Feature Magnitudes", pad=20)
    for i, v in enumerate(mags):
        ax4.text(v, i, f"{v:.3f}", va="center", ha="left")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


# ---------------- Main ----------------

def main():
    dataset_a_images = "/home/joyce/nnUNet/difference/A"
    dataset_a_masks = "/home/joyce/nnUNet/difference/Amask"
    dataset_b_images = "/home/joyce/nnUNet/difference/B"
    dataset_b_masks = "/home/joyce/nnUNet/difference/Bmask"

    calculator = DatasetRatioCalculator()

    print("Loading datasets...")
    images_a, masks_a = load_images_and_masks(dataset_a_images, dataset_a_masks)
    images_b, masks_b = load_images_and_masks(dataset_b_images, dataset_b_masks)

    print("Cropping to same size...")
    images_a, masks_a, images_b, masks_b = crop_images_and_masks(images_a, masks_a, images_b, masks_b)

    print("\nComputing category distribution bias...")
    cat_bias_info, cat_ratios = calculate_category_bias(masks_a, masks_b, calculator.class_weights)

    print("\nComputing pixel statistics...")
    mean_a, mean_b, std_a, std_b = calculate_pixel_stats(images_a, images_b)

    print("\nComputing GLCM contrast...")
    glcm_a = calculate_glcm_contrast(images_a)
    glcm_b = calculate_glcm_contrast(images_b)

    print("\n=== Basic Stats ===")
    print(f"Mean: A={mean_a:.4f}, B={mean_b:.4f}, diff(A-B)={mean_a-mean_b:+.4f}")
    print(f"Std : A={std_a:.4f},  B={std_b:.4f},  diff(A-B)={std_a-std_b:+.4f}")
    print(f"GLCM contrast: A={glcm_a:.4f}, B={glcm_b:.4f}, diff(A-B)={glcm_a-glcm_b:+.4f}")
    print(f"Category bias (abs sum)   : {cat_bias_info['weighted_abs_bias_sum']:.6f}")
    print(f"Category bias (signed sum): {cat_bias_info['weighted_signed_bias_sum']:+.6f}")

    bias_details = {
        "category_bias": cat_bias_info,
        # SIGNED differences (direction matters!)
        "pixel_mean_diff": (mean_a - mean_b),
        "pixel_std_diff": (std_a - std_b),
        "glcm_contrast_diff": (glcm_a - glcm_b),
    }

    print("\nComputing dataset ratios...")
    results = calculator.calculate_dataset_ratios(bias_details)

    print("\n=== Dataset Ratio Results ===")
    print(f"Dataset A ratio : {results['ratio_a']:.4f}")
    print(f"Dataset B ratio : {results['ratio_b']:.4f}")
    print(f"Shift           : {results['shift']:+.5f}  (positive => favor A, negative => favor B)")
    print(f"Direction gate  : {results['direction_gate']:+.4f}")
    print(f"Magnitude gate  : {results['magnitude_gate']:.4f}")
    print(f"Total difference: {results['total_difference']:.4f}")
    print(f"Confidence      : {results['confidence']:.4f}")

    print("\n=== Normalized Magnitudes ===")
    for k, v in results["normalized_magnitude"].items():
        print(f"{k:>22s}: {v:.4f}")

    print("\n=== Normalized Directions ===")
    for k, v in results["normalized_direction"].items():
        print(f"{k:>22s}: {v:+.4f}")

    print("\n=== Importance Scores ===")
    for k, v in results["importance_scores"].items():
        print(f"{k:>22s}: {v:.4f}")

    print("\nGenerating visualizations...")
    plot_feature_importance(results["importance_scores"], results["normalized_magnitude"], save_path="feature_importance.png")
    plot_combined_analysis(results, save_path="combined_analysis.png")
    print("Saved: feature_importance.png")
    print("Saved: combined_analysis.png")


if __name__ == "__main__":
    main()
