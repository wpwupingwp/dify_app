import json
from pathlib import Path
from shutil import move

import torch
from imagededup.methods import PHash
from loguru import logger as log
from torchvision.io import read_image, write_png
from torchvision.transforms import v2 as transforms

big_image_dir = Path(r'F:\IBCAS\SAM\Rosaceae_img').absolute()

TARGET_SIZE = 512
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = 'cpu'
    log.critical('CUDA not available')


def loader(folder: Path, pattern='*.jpg') -> tuple[Path, torch.Tensor]:
    for img_file in folder.rglob(pattern):
        # data = read_image(filename).to(DEVICE)
        data = read_image(img_file)
        yield img_file, data


def resize_image(input_folder: Path):
    input_folder = input_folder.absolute()
    log.info(f'Resizing {input_folder} to {TARGET_SIZE}')
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
        move(filename.absolute(), subfolder/filename.name)
        n += 1
    log.info(f'Moved {n} files')
    if len(subfolders) == 0:
        subfolders = [i for i in folder.glob('*') if i.is_dir()]
        log.info(f'Load {len(subfolders)} subfolders')
    return subfolders


def deduplicate(input_folder: Path) -> list[Path]:
    hasher = PHash()
    encodings = hasher.encode_images(image_dir=input_folder)
    duplicates = hasher.find_duplicates(encoding_map=encodings)
    with open(input_folder / 'duplicates.json', 'w') as f:
        json.dump(duplicates, f)
    print(duplicates)
    return
    # plot duplicates obtained for a given file using the duplicates dictionary
    # from imagededup.utils import plot_duplicates
    # plot_duplicates(image_dir='path/to/image/directory',
    #                 duplicate_map=duplicates,
    #                 filename='ukbench00120.jpg')
    # pass


if __name__ == '__main__':
    subfolders = create_subfolders(big_image_dir)
    resize_image(big_image_dir)
    raise Exception
    for subfolder in subfolders:
        log.info(f'Processing {subfolder}')
        deduplicate(subfolder)
        raise Exception
