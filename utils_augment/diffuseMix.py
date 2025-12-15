import os
from torch.utils.data import Dataset
from PIL import Image
import random
from augment.utils import Utils


class DiffuseMix(Dataset):
    def __init__(self, original_dataset, num_images, guidance_scale, fractal_imgs, idx_to_class, prompts, model_handler, out_dir, part):
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.combine_counter = 0
        self.fractal_imgs = fractal_imgs
        self.prompts = prompts
        self.model_handler = model_handler
        self.num_augmented_images_per_image = num_images
        self.guidance_scale = guidance_scale
        self.utils = Utils()
        self.out_dir = out_dir
        self.part = int(part)
        self.augmented_images = self.generate_augmented_images()

    def generate_augmented_images(self):
        augmented_data = []

        size_of_sample = len(self.original_dataset.samples)

        quarter = size_of_sample // 4
        if 0 <= self.part <= 3:
            start_idx = quarter * self.part
            end_idx   = quarter * (self.part + 1) if self.part < 3 else size_of_sample
        else:
            start_idx, end_idx = 0, size_of_sample

        print("-------------------------------------")
        print("Total :", size_of_sample)
        print("Start Idx :", start_idx)
        print("End Idx :", end_idx - 1)

        for idx, (img_path, label_idx) in enumerate(self.original_dataset.samples):
            if not (start_idx <= idx < end_idx):
                continue

            label = self.idx_to_class[label_idx]           # ex) 727-200
            img_stem = os.path.splitext(os.path.basename(img_path))[0]

            class_dir = os.path.join(self.out_dir, label)
            os.makedirs(class_dir, exist_ok=True)

            original_img = Image.open(img_path).convert('RGB').resize((640, 640))

            prompt = random.choice(self.prompts)
            gen_images = self.model_handler.generate_images(
                prompt, img_path, self.num_augmented_images_per_image, self.guidance_scale
            )

            for i, gimg in enumerate(gen_images):
                if self.utils.is_black_image(gimg):
                    continue

                combined_img = self.utils.combine_images(original_img, gimg)
                random_fractal_img = random.choice(self.fractal_imgs)
                blended_img = self.utils.blend_images_with_resize(combined_img, random_fractal_img)

                save_name = f"{img_stem}_dmix_{i}.jpg"
                outpath = os.path.join(class_dir, save_name)
                blended_img.save(outpath)

                augmented_data.append((blended_img, label))

        if not augmented_data:
            print("[DiffuseMix] No augmented samples produced in this part.")

        return augmented_data
            
    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image, label = self.augmented_images[idx]
        return image, label
