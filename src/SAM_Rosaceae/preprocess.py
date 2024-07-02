import os
from pathlib import Path
from shutil import move

import torch
from loguru import logger as log
from torchvision.transforms import v2 as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.io import read_image, write_png

big_image_dir = Path(r'F:\IBCAS\SAM\Rosaceae_img')

TARGET_SIZE = 512
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = 'cpu'
    log.critical('CUDA not available')
log.info(f'Using {DEVICE}')


def loader(folder: Path) -> tuple[Path, torch.Tensor]:
    for subfolder in folder.glob('*'):
        if subfolder.is_dir():
            for filename in folder.glob('*.jpg'):
                # data = read_image(filename).to(DEVICE)
                data = read_image(filename)
                yield filename, data


def resize_image(input_folder: Path):
    input_folder = input_folder.absolute()
    output_folder = input_folder.parent / (input_folder.name + '-512')
    output_folder = output_folder.absolute()
    if output_folder.exists():
        log.warning('Output directory already exists!')
    output_folder.mkdir(exist_ok=True)
    log.info(f'Output folder: {output_folder}')

    # resize_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(TARGET_SIZE),
    #     transforms.PILToTensor(),
    # ])

    for name, image in loader(input_folder):
        # resized_image = resize_transform(image)
        resized_image = transforms.Resize(TARGET_SIZE)(image)
        log.info(f'Resize from {image.shape} to {resized_image.shape}')
        out_file = output_folder / name.with_suffix('.png').name
        log.info(out_file)
        write_png(resized_image, str(output_folder/name.name))
    return output_folder

    # dataset = ImageFolder(image_directory, transform=resize_transform)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


def create_subfolders(folder: Path) -> list[Path]:
    n = 0
    subfolders = list()
    for filename in folder.glob('*.jpg'):
        name = filename.name.split('+')
        species_name = ' '.join(name[:2])
        subfolder = folder / species_name
        subfolders.append(subfolder)
        if not subfolder.exists():
            subfolder.mkdir()
            log.info(f'Created subfolder: {subfolder}')
        move(filename, subfolder/filename)
        n += 1
    log.info(f'Moved {n} files')
    return subfolders


def deduplicate(input_folder: Path) -> list[Path]:
    from imagededup.methods import PHash
    phasher = PHash()

    # Generate encodings for all images in an image directory
    encodings = phasher.encode_images(image_dir='path/to/image/directory')

    # Find duplicates using the generated encodings
    duplicates = phasher.find_duplicates(encoding_map=encodings)

    # plot duplicates obtained for a given file using the duplicates dictionary
    from imagededup.utils import plot_duplicates
    plot_duplicates(image_dir='path/to/image/directory',
                    duplicate_map=duplicates,
                    filename='ukbench00120.jpg')
    pass


if __name__ == '__main__':
    subfolders = create_subfolders(big_image_dir)
    for subfolder in subfolders:
        resize_image(subfolder)
