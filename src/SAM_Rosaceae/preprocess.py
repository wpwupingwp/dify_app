import json
from pathlib import Path
from shutil import move
from concurrent.futures import ProcessPoolExecutor

import torch
from imagededup.methods import PHash
from imagededup.utils import plot_duplicates
from loguru import logger as log
from torchvision.io import read_image, write_png
from torchvision.transforms import v2 as transforms

from matplotlib import pyplot as plt
from PIL import Image

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


def deduplicate(input_folder: Path) -> Path:
    log.info(f'Processing {subfolder}')
    json_file = input_folder / 'duplicates.json'
    if json_file.exists():
        log.warning(f'Found {json_file}, reload')
    else:
        hasher = PHash()
        encodings = hasher.encode_images(image_dir=input_folder)
        duplicates = hasher.find_duplicates(encoding_map=encodings)
        with open(input_folder / 'duplicates.json', 'w') as f:
            json.dump(duplicates, f, indent=4)
    return json_file


def plot_images(image_dir: Path, orig: str, image_list: list, outfile: Path,
                scores: bool = False) -> Path:
    """
    Plotting function for plot_duplicates() defined below.

    Args:
        image_dir: image directory where all files in duplicate_map are present.
        orig: filename for which duplicates are to be plotted.
        image_list: List of duplicate filenames, could also be with scores (filename, score).
        scores: Whether only filenames are present in the image_list or scores as well.
        outfile:  Name of the file to save the plot.
    """

    def simple_title(title: str):
        # ppbc filename format
        return title.split('_')[3]

    fig = plt.figure(figsize=(5*len(image_list), 10))
    _, ax_list = plt.subplots(nrows=1, ncols=len(image_list)+1)
    ax = ax_list[0]
    ax.imshow(Image.open(image_dir / orig))
    ax.set_title(f'Original: {simple_title(orig)}')
    for i, ax in enumerate(ax_list[1:]):
        if scores:
            ax.imshow(Image.open(image_dir / image_list[i][0]))
            title = f'{simple_title(image_list[i][0])} ({image_list[i][1]:.2%}'
        else:
            ax.imshow(Image.open(image_dir / image_list[i]))
            title = simple_title(image_list[i])
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    plt.savefig(outfile)
    plt.close()
    return outfile


def parse_duplicate(info_json: Path) -> Path:
    to_delete = []
    duplicates = json.loads(info_json.read_text())
    n_raw = len(duplicates)
    for k, v in duplicates.items():
        for _ in v:
            if _ in duplicates:
                duplicates[_] = []
    duplicates = {k: v for k, v in duplicates.items() if v}
    log.info(f'Found {len(duplicates)} pairs of duplicates in {n_raw}')
    for k, v in duplicates.items():
        if isinstance(v, (str, Path)):
            v = [v, ]
        to_delete.extend([k, *v, ''])
    result = info_json.with_name('to_delete.json')
    result.write_text(json.dumps(to_delete, indent=True, ensure_ascii=False))

    image_folder = info_json.parent
    for image in duplicates:
        retrieved = duplicates[image]
        outfile = image_folder / ('result-' + Path(image).name)
        if isinstance(duplicates[image], tuple):
            plot_images(image_dir=image_folder, orig=image,
                        image_list=retrieved, scores=True, outfile=outfile, )
        else:
            plot_images(image_dir=image_folder, orig=image,
                        image_list=retrieved, scores=False, outfile=outfile, )
    return result


if __name__ == '__main__':
    log.info(f'Input directory: {input_directory}')
    subfolders = create_subfolders(input_directory)
    resized_dir = input_directory.parent / 'resized'
    log.info(f'Resized directory: {resized_dir}')
    resized_dir.mkdir(exist_ok=True)
    with ProcessPoolExecutor() as executor:
        for i in subfolders:
            resized_folder = resized_dir / i.name
            if not resized_folder.exists():
                log.info(f'Create resized subfolder: {resized_folder}')
                resized_folder.mkdir()
                executor.submit(resize_image, i, resized_folder)
            else:
                log.warning(f'Found {resized_folder}, skip')
    log.info('Resize finished')
    log.info('Start deduplicating')
    duplicate_info = list()
    for subfolder in resized_dir.glob('*'):
        if not subfolder.is_dir():
            continue
        duplicate_info.append(deduplicate(subfolder))
    for info_json in duplicate_info:
        parse_duplicate(info_json)
    log.info('Deduplicate finished')
