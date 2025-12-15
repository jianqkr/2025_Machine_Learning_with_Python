import os
from torch.utils.data import Dataset
from PIL import Image
import random
from bbox_utils.utils import Utils


class DiffuseMix(Dataset):
    def __init__(
        self,
        original_dataset,
        num_images,
        guidance_scale,
        fractal_imgs,
        idx_to_class,
        prompts,
        model_handler,
        out_dir,
        part,
        bbox_dict=None
    ):
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.fractal_imgs = fractal_imgs or []
        self.prompts = prompts or []
        self.model_handler = model_handler
        self.num_augmented_images_per_image = int(num_images)
        self.guidance_scale = guidance_scale
        self.utils = Utils()
        self.out_dir = out_dir
        self.part = int(part)
        self.bbox_dict = bbox_dict or {}  # key: "1025794", value: (xmin, ymin, xmax, ymax)

        if not self.fractal_imgs:
            raise FileNotFoundError(
                "No fractal images provided/loaded. Put images in fractal_dir and pass them in."
            )
        if not self.prompts:
            raise ValueError("No prompts provided.")

        self.augmented_images = self.generate_augmented_images()

    def generate_augmented_images(self):
        augmented_data = []

        size_of_sample = len(self.original_dataset.samples)

        quarter = size_of_sample // 4
        if 0 <= self.part <= 3:
            start_idx = quarter * self.part
            end_idx = quarter * (self.part + 1) if self.part < 3 else size_of_sample
        else:
            start_idx, end_idx = 0, size_of_sample

        print("-------------------------------------")
        print("Total :", size_of_sample)
        print("Start Idx :", start_idx)
        print("End Idx :", end_idx - 1)

        for idx, (img_path, label_idx) in enumerate(self.original_dataset.samples):
            if not (start_idx <= idx < end_idx):
                continue

            label = self.idx_to_class[label_idx] 
            img_stem = os.path.splitext(os.path.basename(img_path))[0]

            class_dir = os.path.join(self.out_dir, label)
            os.makedirs(class_dir, exist_ok=True)

            full_img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = full_img.size

            resized_img = full_img.resize((640, 640))
            bbox = self.bbox_dict.get(img_stem, None)

            prompt = random.choice(self.prompts)

            try:
                gen_images = self.model_handler.generate_images(
                    prompt,
                    resized_img,  
                    self.num_augmented_images_per_image,
                    self.guidance_scale
                )
            except Exception as e:
                print(f"[DiffuseMix] Generation failed at idx {idx} ({img_path}): {e}")
                continue

            for i, gen_full in enumerate(gen_images):
                if bbox is None:
                    if self.utils.is_black_image(gen_full):
                        continue

                    combined_img = self.utils.combine_images(resized_img, gen_full)
                    random_fractal_img = random.choice(self.fractal_imgs)
                    blended_img = self.utils.blend_images_with_resize(
                        combined_img,
                        random_fractal_img
                    )

                    save_name = f"{img_stem}_dmix_{i}.jpg"
                    outpath = os.path.join(class_dir, save_name)
                    blended_img.save(outpath)
                    augmented_data.append((blended_img, label))
                    continue

                xmin, ymin, xmax, ymax = bbox  # images_box.txt

                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(orig_w, xmax)
                ymax = min(orig_h, ymax)

                if xmax <= xmin or ymax <= ymin:
                    print(f"[WARN] Invalid bbox for {img_stem}: {bbox}. Using full image instead.")
                    if self.utils.is_black_image(gen_full):
                        continue
                    combined_img = self.utils.combine_images(resized_img, gen_full)
                    random_fractal_img = random.choice(self.fractal_imgs)
                    blended_img = self.utils.blend_images_with_resize(
                        combined_img,
                        random_fractal_img
                    )
                    save_name = f"{img_stem}_dmix_{i}.jpg"
                    outpath = os.path.join(class_dir, save_name)
                    blended_img.save(outpath)
                    augmented_data.append((blended_img, label))
                    continue

                sx = 640.0 / orig_w
                sy = 640.0 / orig_h

                xmin_r = int(round(xmin * sx))
                ymin_r = int(round(ymin * sy))
                xmax_r = int(round(xmax * sx))
                ymax_r = int(round(ymax * sy))

                xmin_r = max(0, min(639, xmin_r))
                ymin_r = max(0, min(639, ymin_r))
                xmax_r = max(xmin_r + 1, min(640, xmax_r))
                ymax_r = max(ymin_r + 1, min(640, ymax_r))

                if xmax_r - xmin_r < 2 or ymax_r - ymin_r < 2:
                    print(f"[WARN] Very small bbox for {img_stem}: {bbox}. Using full image instead.")
                    if self.utils.is_black_image(gen_full):
                        continue
                    combined_img = self.utils.combine_images(resized_img, gen_full)
                    random_fractal_img = random.choice(self.fractal_imgs)
                    blended_img = self.utils.blend_images_with_resize(
                        combined_img,
                        random_fractal_img
                    )
                    save_name = f"{img_stem}_dmix_{i}.jpg"
                    outpath = os.path.join(class_dir, save_name)
                    blended_img.save(outpath)
                    augmented_data.append((blended_img, label))
                    continue

                orig_patch = resized_img.crop((xmin_r, ymin_r, xmax_r, ymax_r))
                gen_patch = gen_full.crop((xmin_r, ymin_r, xmax_r, ymax_r))

                if self.utils.is_black_image(gen_patch):
                    continue

                combined_patch = self.utils.combine_images(orig_patch, gen_patch)
                augmented_full = resized_img.copy()
                augmented_full.paste(combined_patch, (xmin_r, ymin_r))

                random_fractal_img = random.choice(self.fractal_imgs)
                final_blended = self.utils.blend_images_with_resize(
                    augmented_full,
                    random_fractal_img
                )

                save_name = f"{img_stem}_dmix_{i}.jpg"
                outpath = os.path.join(class_dir, save_name)
                final_blended.save(outpath)

                augmented_data.append((final_blended, label))

        if not augmented_data:
            print("[DiffuseMix] No augmented samples produced in this part.")

        return augmented_data

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image, label = self.augmented_images[idx]
        return image, label
