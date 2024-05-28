from loguru import logger as log
from pathlib import Path
from torchmetrics.multimodal import CLIPImageQualityAssessment as clip_iqa
from shutil import copy, rmtree

import torchvision

model = 'openai/clip-vit-base-patch32'
# model = 'openai/clip-vit-large-patch14'
prompts = [('Good photo.', 'Bad photo.'),
           ('Complex photo.', 'Simple photo.'),
           ('Beautiful photo.', 'Ugly photo.')
           ('Aesthetic photo.', 'Not aesthetic photo.'),
           ('Plant photo.', 'not plant photo.') ]
output_file = open('result.txt', 'w')
output_folder = Path('/tmp/out')

def scan_files(workdir=None) -> Path:
    count = 0
    PAUSE = 100
    if workdir is None:
        workdir = '/media/ping/Data/Work/pics'
    for folder in Path(workdir).glob('*'):
        for img in folder.glob('*'):
            if img.suffix in {'.jpg', '.png', '.jpeg'}:
                yield img
                count += 1
                if count > PAUSE:
                    pass


def analyze(img_score: list, output_folder: Path):
    img_score.sort(key=lambda x:float(x[1][0]))
    last_score = 'x'
    count = 0
    for r in img_score:
        img, score, *_ = r
        s1 = str(score)[2]
        if s1 != last_score:
            count = 0
            last_score = s1
        if count > 9:
            continue
        p = Path(img)
        copy(img, output_folder/(s1+'-'+p.name))
        count += 1
        log.info(r)


def main():
    log.add('iqa_{time}.log')
    log.info('Start')
    m = clip_iqa(prompts=prompts, model_name_or_path=model).to('cuda')
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
    for n, img_file in enumerate(scan_files()):
        log.info(f'{n} {img_file}')
        # unsqueeze to 4-d tensor 
        img = torchvision.io.read_image(img_file).unsqueeze(0).to('cuda')
        s = m(img)
        values = s.values()
        values_str = ','.join([f'{i.item():.2f}' for i in values])
        output_file.write(f'{img_file},{values_str}\n')

        img_score.append((img_file, values))
    log.info('Sorting score')
    analyze(img_score, output_folder)
    log.info('Done')


main()