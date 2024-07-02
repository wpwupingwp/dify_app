from loguru import logger as log
from pathlib import Path
from torchmetrics.multimodal import CLIPImageQualityAssessment as clip_iqa
from shutil import copy, rmtree

import torchvision

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
    for n, img_file in enumerate(scan_files(input_folder)):
        log.info(f'{n} {img_file}')
        # unsqueeze to 4-d tensor
        img = torchvision.io.read_image(img_file).unsqueeze(0).to('cuda')
        s = m(img)
        values = [i.item() for i in s.values()]
        values_str = ','.join([f'{i:.2f}' for i in values])
        output_file.write(f'{img_file},{values_str}\n')

        img_score.append((img_file, values))
    log.info('Sorting score')
    analyze(img_score, output_folder)
    log.info('Done')


main()
