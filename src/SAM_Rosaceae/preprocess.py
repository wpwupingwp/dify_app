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
            yield filename.absolute()


def parse_ppbc_name(name: Path) -> tuple[str, str, str, str] | None:
    # ppbc name: Prunus+armeniaca+L._杏_PPBC_2712691_朱鑫鑫_河南省_信阳市浉河区.jpg
    fields = name.name.split('_')
    if len(fields) >= 3:
        species_name, chinese_name, _, name_id = fields[:4]
    else:
        return None
    suffix = name.suffix
    return species_name, chinese_name, name_id, suffix


def fake_delete(img: Path, name_dict: dict, dest: Path) -> Path:
    log.info(f'Fake-delete {img}')
    # move original file to deleted folder according to resized scored filename
    original_name, folder_name = name_dict[img.name]
    original_name = Path(original_name)
    folder_name = Path(folder_name)
    old_name = original_name.parent / folder_name.name / img.name
    new_folder = dest / folder_name.name
    if not new_folder.exists():
        new_folder.mkdir()
    new_name = new_folder / img.name
    try:
        move(old_name, new_name)
    except FileNotFoundError:
        log.error(f'Failed to delete {img}')
        return img
    return new_name


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
        out_file = (output_folder / filename.name)
        write_png(resized_image, str(out_file))
    return output_folder


def organize(folder: Path, name_file: Path) -> tuple[list[Path], dict]:
    # rename files, create subfolders, move files
    name_dict = dict()
    n = 0
    subfolders = list()
    for filename in list(loader(folder)):
        fields = parse_ppbc_name(filename)
        if fields is None:
            continue
        species_name, _, name_id, suffix = fields
        subfolder = folder / species_name
        if not subfolder.exists():
            subfolders.append(subfolder)
            subfolder.mkdir()
            log.info(f'Created subfolder: {subfolder}')
        # label studio only accept simple filename
        new_name = subfolder / (name_id+suffix)
        name_dict[new_name.name] = (str(filename), str(subfolder))
        move(filename, new_name)
        n += 1
    log.info(f'Moved {n} files')
    if len(subfolders) == 0:
        subfolders = [i for i in folder.glob('*') if i.is_dir()]
        log.info(f'Load {len(subfolders)} subfolders')
    name_file.write_text(json.dumps(name_dict, indent=4, ensure_ascii=False),
                         encoding='utf-8')
    return subfolders, name_dict


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
            json.dump(duplicates, f, indent=4, ensure_ascii=False)
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

    fig = plt.figure(figsize=(5*len(image_list), 10))
    _, ax_list = plt.subplots(nrows=1, ncols=len(image_list)+1)
    ax = ax_list[0]
    ax.axis('off')
    ax.imshow(Image.open(image_dir / orig))
    ax.set_title(f'Original: {orig}')
    for i, ax in enumerate(ax_list[1:]):
        if scores:
            ax.imshow(Image.open(image_dir / image_list[i][0]))
            title = f'{image_list[i][0]} ({image_list[i][1]:.2%}'
        else:
            ax.imshow(Image.open(image_dir / image_list[i]))
            title = image_list[i]
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
    result.write_text(json.dumps(to_delete, indent=4, ensure_ascii=False),
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


def score(input_folder: Path, low_score=0.25) -> Path:
    """
    use clip-iqa to filter low score images
    Args:
        input_folder:
        low_score: score threshold
    Returns:
        low_score_json: Path
    """
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
    if low_score_json.exists():
        log.warning(f'Found {low_score_json}, reload')
        return low_score_json
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
            low_score_files.append(str(img_file))
        img_score.append([str(img_file), values])
    score_json.write_text(json.dumps({'prompts': prompts, 'score': img_score},
                          indent=4), encoding='utf-8')
    low_score_json.write_text(json.dumps(low_score_files, indent=4),
                              encoding='utf-8')
    log.info(f'Found {len(low_score_files)} low-score images in {input_folder}')
    return low_score_json


def main(input_directory: Path):
    log.info(f'Input directory: {input_directory}')
    log.info(f'Organizing files ')
    name_file = input_directory / 'name_info.json'
    subfolders, name_dict = organize(input_directory, name_file)
    log.info(f'See {name_file} for organize history')

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
    duplicate_list = list()
    for info_json in duplicate_info:
        duplicates = parse_duplicate(info_json)
        duplicate_list.extend(json.loads(duplicates.read_text(
            encoding='utf-8')))
    log.info(f'Found {len(duplicate_list)} duplicated images should be removed')
    log.info('Deduplicate finished')

    log.info('Start scoring')
    low_score_list = list()
    for subfolder in resized_dir.glob('*'):
        if subfolder.is_dir():
            low_score_json = score(subfolder)
            low_score_list.extend(json.loads(
                low_score_json.read_text(encoding='utf-8')))
    log.info(f'Found {len(low_score_list)} low quality images')

    log.info('Fake-delete low quality or duplicated images')
    delete_folder = input_directory.parent / 'delete'
    delete_folder.mkdir()
    log.info(f'Move them to {delete_folder}')
    for i in duplicate_list:
        fake_delete(Path(i), name_dict, delete_folder)
    for j in low_score_list:
        fake_delete(Path(j), name_dict, delete_folder)
    log.info(f'Fake-deleted {len(low_score_list)+len(duplicate_list)} images')
    log.info('Done')


if __name__ == '__main__':
    # input_directory = Path(r'F:\IBCAS\SAM\Rosaceae').absolute()
    input_directory = Path(r'/media/ping/Data/Work/Rosaceae').absolute()
    main(input_directory)
