import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import move

import torch
from PIL import Image
from imagededup.methods import PHash
from loguru import logger as log
from matplotlib import pyplot as plt
from torchmetrics.multimodal import CLIPImageQualityAssessment as clip_iqa
from torchvision.io import read_image, write_png
from torchvision.transforms import v2 as transforms

TARGET_SIZE = 512
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = 'cpu'
    log.critical('CUDA not available')


def loader(folder: Path) -> Iterable[Path]:
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
    log.info(f'Processing {input_folder}')
    json_file = input_folder / 'duplicates.json'
    if json_file.exists():
        log.warning(f'Found {json_file}, reload')
    else:
        hasher = PHash()
        encodings = hasher.encode_images(image_dir=input_folder)
        duplicates = hasher.find_duplicates(encoding_map=encodings)
        with open(input_folder / 'duplicates.json', 'w', encoding='utf-8') as f:
            json.dump(duplicates, f, indent=True, ensure_ascii=False)
    return json_file


def plot_images(image_dir: Path, orig: str, image_list: list, outfile: Path,
                scores: bool = False) -> Path:
    """
    Plotting function for plot_duplicates() defined below.

    Args:
        image_dir: image directory where all files in duplicate_map are present.
        orig: filename for which duplicates are to be plotted.
        image_list: List of duplicate filenames, or tuple (filename, score).
        scores: Whether only filenames are present in the image_list or scores
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
    result.write_text(json.dumps(to_delete, indent=True, ensure_ascii=False),
                      encoding='utf-8')

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


def score(input_folder: Path, low_score=0.3) -> Path:
    """
    use clip-iqa to filter low score images
    Args:
        input_folder:
        low_score: score threshold
    Returns:
        low_score_json: Path
    """
    model_file = 'openai/clip-vit-large-patch14'
    model_file = 'openai/clip-vit-base-patch32'
    data_range = 1.0
    log.info(f'Use {low_score} as low score threshold')
    # clip_iqa use 0-255, clip use 0-1.0
    # data_range = 255
    # model = 'clip_iqa'
    prompts = (('Good photo.', 'Bad photo.'),
               ('Aesthetic photo.', 'Not aesthetic photo.'),
               ('Natural photo.', 'Synthetic photo.'),
               ('Plant photo', 'Not plant photo'),
               ('Flower photo', 'Not flower photo'),
               ('Leaf photo', 'Not leaf photo'))
    model = clip_iqa(prompts=prompts, model_name_or_path=model_file,
                     data_range=data_range).to(DEVICE)
    log.info(f'Loaded {model_file}')
    score_json = input_folder / 'score.json'
    low_score_json = input_folder / 'low_score.json'
    log.info(f'Prompts:{prompts}')
    img_score = list()
    low_score_files = list()

    for img_file in loader(input_folder):
        img = read_image(img_file).unsqueeze(0)
        try:
            scores = model(img)
        except ValueError:
            log.critical(f'Failed to get score of {img_file}')
            continue
        values = [i.item() for i in scores.values()]
        quality = values[0]
        if quality < low_score:
            log.warning(f'Found low score image {img_file.name}: {quality:.3f}')
            low_score_files.append([str(img_file), quality])
        img_score.append([str(img_file), values])
    score_json.write_text(json.dumps({'prompts': prompts, 'score': img_score},
                          indent=True), encoding='utf-8')
    low_score_json.write_text(json.dumps(low_score_files, indent=True),
                              encoding='utf-8')
    log.info(f'Found {len(low_score_files)} in {input_folder}')
    return low_score_json


def main(input_directory: Path):
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
    log.info('Start scoring')
    low_score_img = list()
    for subfolder in resized_dir.glob('*'):
        if subfolder.is_dir():
            low_score_json = score(subfolder)
            low_score_img.extend(json.loads(
                low_score_json.read_text(encoding='utf-8')))
    log.info('Delete/move low quality images')
    for i in to_delete_list:
        pass
    for i in low_score_img:
        pass
    log.info('Done')


if __name__ == '__main__':
    input_directory = Path(r'F:\IBCAS\SAM\Rosaceae_img').absolute()
    main(input_directory)
