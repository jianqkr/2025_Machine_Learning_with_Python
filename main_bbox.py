import argparse
from torchvision import datasets
from utils_bbox.handler import ModelHandler
from utils_bbox.utils import Utils
from utils_bbox.diffuseMix import DiffuseMix

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate an augmented dataset from original images and fractal patterns (bbox-aware)."
    )
    parser.add_argument(
        '--train_dir', type=str, required=True,
        help='Path to the directory containing the original training images.'
    )
    parser.add_argument(
        '--fractal_dir', type=str, required=True,
        help='Path to the directory containing the fractal images.'
    )
    parser.add_argument(
        '--out_dir', type=str, required=True,
        help='Path to the directory where the augmented images will be saved.'
    )
    parser.add_argument(
        '--device', type=str, required=True,
        help='CUDA device index, e.g. 0,1,2,3'
    )
    parser.add_argument(
        '--bbox_file', type=str, required=True,
        help='Path to images_box.txt containing bounding boxes.'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Initialize the model
    model_id = "timbrooks/instruct-pix2pix"
    model_initialization = ModelHandler(
        model_id=model_id,
        device=f'cuda:{args.device}'
    )

    # Load the original dataset
    train_dataset = datasets.ImageFolder(root=args.train_dir)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Load fractal images
    fractal_imgs = Utils.load_fractal_images(args.fractal_dir)

    # Load bounding boxes
    bbox_dict = Utils.load_bboxes(args.bbox_file)

    # Create the augmented dataset (bbox-aware DiffuseMix)
    augmented_train_dataset = DiffuseMix(
        original_dataset=train_dataset,
        fractal_imgs=fractal_imgs,
        num_images=1,
        guidance_scale=4,
        idx_to_class=idx_to_class,
        
        prompts=[
            "Autumn", "snowy", "watercolor art", "sunset", "rainbow", "aurora",
            "mosaic", "ukiyo-e", "a sketch with crayon"
        ],

        # prompts= [
        #     "Pointillism", "sepia", "etching", "twilight", "grainy film texture",
        #     "spring bloom", "stormy sky", "pixel art", "golden hour"
        # ],
        model_handler=model_initialization,
        out_dir=args.out_dir,
        part=args.device,
        bbox_dict=bbox_dict
    )

if __name__ == '__main__':
    main()
