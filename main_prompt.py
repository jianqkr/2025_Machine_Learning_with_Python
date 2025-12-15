import argparse
from torchvision import datasets
from utils_augment.handler import ModelHandler
from utils_augment.utils import Utils
from utils_augment.diffuseMix import DiffuseMix


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an augmented dataset from original images and fractal patterns.")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the directory containing the original training images.')
    parser.add_argument('--fractal_dir', type=str, required=True, help='Path to the directory containing the fractal images.')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to the directory where the augmented images will be saved.')
    parser.add_argument('--device', type=str, required=True)
    
    return parser.parse_args()

def main():

    args = parse_arguments()

    # Initialize the model
    model_id = "timbrooks/instruct-pix2pix"
    model_initialization = ModelHandler(model_id=model_id, device=f'cuda:{args.device}')

    # Load the original dataset
    train_dataset = datasets.ImageFolder(root=args.train_dir)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Load fractal images
    fractal_imgs = Utils.load_fractal_images(args.fractal_dir)

    # Create the augmented dataset
    augmented_train_dataset = DiffuseMix(
        original_dataset=train_dataset,
        fractal_imgs=fractal_imgs,
        num_images=1,
        guidance_scale=4,
        idx_to_class = idx_to_class,
        # prompts=["Autumn", "snowy", "watercolor art","sunset", "rainbow", "aurora",
        #        "mosaic", "ukiyo-e", "a sketch with crayon"],
        prompts= ["Pointillism", "sepia", "etching", "twilight", "grainy film texture",
            "spring bloom", "stormy sky", "pixel art", "golden hour"],
        model_handler=model_initialization,
        out_dir = args.out_dir,
        part = args.device
    )

    # for idx, (image, label) in enumerate(augmented_train_dataset):
    #     image.save(f'augmented_images/{idx}.png')
    #     pass

if __name__ == '__main__':
    main()


