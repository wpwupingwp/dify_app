import os
from pathlib import Path

import torch
from loguru import logger as log
from PIL import Image
from torchvision import transforms

from .config import output as big_image_dir


def batch_resize_images(image_dir: Path, output_dir: Path,
                        target_size=(512, 512), batch_size=32):
    """
    Resizes images in a directory to a given target size and saves them,
    efficiently using CUDA and batch processing.
    WARNING: Keep original width/height ratio!
    Args:
        image_dir (Path):
        output_dir (Path):
        target_size (tuple): Desired size (width, height) of the resized images.
        batch_size (int): Number of images to process in each batch.
    """
    # 1. Define Transformations
    resize_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    # 2. CUDA Check and Device Selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
        log.critical('CUDA not available')
    log.info(f'Using {device}')
    # 3. Process Images in Batches
    # all ends with '.jpg'
    image_filenames = list(image_dir.glob('*.jpg'))
    num_images = len(image_filenames)
    log.info(f'Found {num_images} images')

    for i in range(0, num_images, batch_size):
        batch_filenames = image_filenames[i: i + batch_size]
        batch_images = []

        # Load and Transform Batch on CPU
        for image_path in batch_filenames:
            image = Image.open(image_path).convert("RGB")
            image_tensor = resize_transform(image)
            batch_images.append(image_tensor)

        # Stack into Batch Tensor and Move to GPU
        batch_tensor = torch.stack(batch_images).to(device)

        # (No operations performed on GPU in this example, but you would usually
        # have your model or other processing here)

        # Move Batch back to CPU and Save
        for j, resized_tensor in enumerate(batch_tensor.cpu()):
            resized_image = transforms.ToPILImage()(resized_tensor)
            resized_image.save(os.path.join(output_dir, batch_filenames[j]))


if __name__ == '__main__':
    image_directory = big_image_dir.absolute()
    output_directory = big_image_dir.parent / (big_image_dir.name + '-512x512')
    output_directory = output_directory.absolute()
    if output_directory.exists():
        log.warning('Output directory already exists!')
    output_directory.mkdir(exist_ok=True)
    batch_resize_images(image_directory, output_directory,
                        target_size=(512, 512))
