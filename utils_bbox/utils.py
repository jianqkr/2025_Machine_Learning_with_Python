import os
import random
import numpy as np
from PIL import Image

class Utils:
    @staticmethod
    def load_fractal_images(fractal_img_dir):
        fractal_img_paths = [
            os.path.join(fractal_img_dir, fname)
            for fname in os.listdir(fractal_img_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        return [
            Image.open(path).convert('RGB').resize((640, 640))
            for path in fractal_img_paths
        ]

    @staticmethod
    def load_bboxes(bbox_txt_path):
        bbox_dict = {}
        with open(bbox_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                img_id = parts[0]         
                xmin = int(parts[1])
                ymin = int(parts[2])
                xmax = int(parts[3])
                ymax = int(parts[4])
                bbox_dict[img_id] = (xmin, ymin, xmax, ymax)
        return bbox_dict

    @staticmethod
    def blend_images_with_resize(base_img, overlay_img, alpha=0.20):
        overlay_img_resized = overlay_img.resize(base_img.size)
        base_array = np.array(base_img, dtype=np.float32)
        overlay_array = np.array(overlay_img_resized, dtype=np.float32)
        assert base_array.shape == overlay_array.shape and len(base_array.shape) == 3
        blended_array = (1 - alpha) * base_array + alpha * overlay_array
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
        blended_img = Image.fromarray(blended_array)
        return blended_img

    @staticmethod
    def combine_images(original_img, augmented_img, blend_width=20):
        if augmented_img.size != original_img.size:
            augmented_img = augmented_img.resize(original_img.size)

        width, height = original_img.size

        blend_width = max(1, min(blend_width, min(width, height)))
        original_array = np.array(original_img, dtype=np.float32) / 255.0
        augmented_array = np.array(augmented_img, dtype=np.float32) / 255.0

        assert original_array.shape == augmented_array.shape, \
            f"Shape mismatch: {original_array.shape} vs {augmented_array.shape}"

        combine_choice = random.choice(['horizontal', 'vertical'])
        mask_2d = np.zeros((height, width), dtype=np.float32)

        if combine_choice == 'vertical':
            center = height // 2
            half_bw = blend_width // 2

            start = max(0, center - half_bw)
            end = min(height, start + blend_width)
            actual_bw = max(1, end - start)

            mask_2d[end:, :] = 1.0

            ramp = np.linspace(0.0, 1.0, actual_bw, endpoint=True).reshape(actual_bw, 1)
            mask_2d[start:end, :] = ramp

        else:  # 'horizontal'
            center = width // 2
            half_bw = blend_width // 2

            start = max(0, center - half_bw)
            end = min(width, start + blend_width)
            actual_bw = max(1, end - start)

            mask_2d[:, end:] = 1.0

            ramp = np.linspace(0.0, 1.0, actual_bw, endpoint=True).reshape(1, actual_bw)
            mask_2d[:, start:end] = ramp

        mask = np.repeat(mask_2d[:, :, np.newaxis], 3, axis=2)

        blended_array = (1.0 - mask) * original_array + mask * augmented_array
        blended_array = np.clip(blended_array * 255.0, 0, 255).astype(np.uint8)
        blended_img = Image.fromarray(blended_array)
        return blended_img

    @staticmethod
    def is_black_image(image):
        histogram = image.convert("L").histogram()
        return (
            histogram[-1] > 0.9 * image.size[0] * image.size[1]
            and max(histogram[:-1]) < 0.1 * image.size[0] * image.size[1]
        )
