import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import copy, rmtree
from shutil import move

import torch
import torchvision
from PIL import Image
from imagededup.methods import PHash
from loguru import logger as log
from matplotlib import pyplot as plt
from torchmetrics.multimodal import CLIPImageQualityAssessment as clip_iqa
from torchvision.io import read_image, write_png
from torchvision.transforms import v2 as transforms

input_directory = Path(r'F:\IBCAS\SAM\Rosaceae_img').absolute()

TARGET_SIZE = 512
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = 'cpu'
    log.critical('CUDA not available')


def loader(folder: Path) -> Path:
    # avoid match new files?
    for filename in list(folder.rglob('*')):
        if filename.suffix in {'.jpg', '.jpeg', '.png'}:
            yield filename


def resize_image(input_folder: Path, output_folder: Path) -> Path:
    # resize_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(TARGET_SIZE),
    #     transforms.PILToTensor(),
    # ])
    # dataset = ImageFolder(image_directory, transform=resize_transform)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for filename in loader(input_folder):
        image = read_image(filename)
        # data = read_image(filename).to(DEVICE)
        resized_image = transforms.Resize(TARGET_SIZE)(image)
        log.info(f'Resize {filename.name} from {image.shape} to '
                 f'{resized_image.shape}')
        out_file = (output_folder / filename.with_suffix('.png').name)
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
    log.info(f'Found {len(duplicates)} pairs of duplicates in {n_raw} '
             f'{info_json.parent}')
    for k, v in duplicates.items():
        if isinstance(v, (str, Path)):
            v = [v, ]
        to_delete.extend(v)
    result = info_json.with_name('to_delete.json')
    result.write_text(json.dumps(to_delete, indent=True, ensure_ascii=False,
                                 encoding='utf-8'))

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


def score():

    # model = 'openai/clip-vit-base-patch32'
    model = 'openai/clip-vit-large-patch14'
    data_range = 1.0
    # clip_iqa use 0-255, clip use 0-1.0
    # data_range = 255
    # model = 'clip_iqa'
    prompts = (('Good photo.', 'Bad photo.'),
               ('Aesthetic photo.', 'Not aesthetic photo.'),
               ('Natural photo.', 'Synthetic photo.'),
               ('Plant photo', 'Not plant photo'),
               ('Flower photo', 'Not flower photo'),
               ('Leaf photo', 'Not leaf photo'))

    # prompts = ('quality', 'brightness', 'noisiness', 'colorfullness', 'sharpness', 'contrast', 'complexity', 'natural', 'beautiful')
    input_folder = Path('/tmp/a')
    input_folder = Path('/media/ping/Data/Work/pics')
    output_file = open('result.txt', 'w')
    output_folder = Path('/tmp/out')

    def scan_files(folder: Path) -> Path:
        count = 0
        PAUSE = 100
        for img in folder.glob('*'):
            if img.suffix in {'.jpg', '.png', '.jpeg'}:
                yield img
                count += 1
                if count > PAUSE:
                    pass


    def analyze(img_score: list, output_folder: Path):
        # prompt index
        index = 0
        # name, [value1, value2, ...]
        img_score.sort(key=lambda x: x[1][index])
        last_score = 'x'
        count = 0
        for r in img_score:
            img, scores = r
            score = scores[index]
            s1 = str(score)[2]
            if s1 != last_score:
                count = 0
                last_score = s1
            if count > 9:
                continue
            p = Path(img)
            new_name = output_folder / (s1+'-'+p.name)
            copy(img, new_name)
            count += 1


    def main():
        # log.add('iqa_{time}.log')
        log.info('Start')
        m = clip_iqa(prompts=prompts, model_name_or_path=model,).to('cuda')
        # m = clip_iqa(prompts=prompts, model_name_or_path=model, data_range=255).to('cuda')
        log.info('Model loaded')
        log.info(f'Prompts:{prompts}')

        output_file.write(str(prompts)+'\n')
        if not output_folder.exists():
            output_folder.mkdir()
        else:
            log.warning('Clean old output')
            rmtree(output_folder)
        log.info(f'Output folder {output_folder}')

        log.info('Analyzing')
        img_score = list()
        for n, img_file in enumerate(loader(input_folder)):
            log.info(f'{n} {img_file}')
            # unsqueeze to 4-d tensor
            img = read_image(img_file).unsqueeze(0).to('cuda')
            s = m(img)
            values = [i.item() for i in s.values()]
            values_str = ','.join([f'{i:.2f}' for i in values])
            output_file.write(f'{img_file},{values_str}\n')

            img_score.append((img_file, values))
        log.info('Sorting score')
        analyze(img_score, output_folder)
        log.info('Done')


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
    to_delete_list = list()
    for info_json in duplicate_info:
        to_delete = parse_duplicate(info_json)
        to_delete_list.extend(json.loads(to_delete.read_text(encoding='utf-8')))
    log.info(f'Found {len(to_delete_list)} duplicated images should be removed')
    log.info('Deduplicate finished')
