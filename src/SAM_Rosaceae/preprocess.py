import json
from pathlib import Path
from shutil import move
from concurrent.futures import ProcessPoolExecutor

import torch
from imagededup.methods import PHash
from loguru import logger as log
from torchvision.io import read_image, write_png
from torchvision.transforms import v2 as transforms

input_directory = Path(r'F:\IBCAS\SAM\Rosaceae_img').absolute()

TARGET_SIZE = 512
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = 'cpu'
    log.critical('CUDA not available')


def loader(folder: Path, pattern='*.jpg') -> tuple[Path, torch.Tensor]:
    # avoid match new files
    for img_file in list(folder.rglob(pattern)):
        # data = read_image(filename).to(DEVICE)
        data = read_image(img_file)
        yield img_file, data


def resize_image(input_folder: Path, output_folder: Path) -> Path:
    # resize_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(TARGET_SIZE),
    #     transforms.PILToTensor(),
    # ])
    # dataset = ImageFolder(image_directory, transform=resize_transform)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for name, image in loader(input_folder):
        resized_image = transforms.Resize(TARGET_SIZE)(image)
        log.info(f'Resize {name.name} from {image.shape} to {resized_image.shape}')
        out_file = (output_folder / name.with_suffix('.png').name)
        write_png(resized_image, str(out_file))
    return output_folder


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
    log.info(f'Input directory: {input_directory}')
    subfolders = create_subfolders(input_directory)
    resized_dir = input_directory.parent / 'resized'
    log.info(f'Resized directory: {resized_dir}')
    resized_dir.mkdir(exist_ok=True)
    with ProcessPoolExecutor() as executor:
        for i in subfolders:
            resized_folder = resized_dir / i.name
            log.info(f'Create resized subfolder: {resized_folder}')
            resized_folder.mkdir(exist_ok=True)
            executor.submit(resize_image, i, resized_folder)

    raise Exception
    for subfolder in resized_folders:
        log.info(f'Processing {subfolder}')
        deduplicate(subfolder)
        raise Exception
