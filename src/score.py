from loguru import logger as log
from pathlib import Path
from torchmetrics.multimodal import CLIPImageQualityAssessment as clip_iqa

import loguru
import torch
import torchvision

def scan_files(workdir=None) -> Path:
    if workdir is None:
        workdir = '/media/ping/Data/Work/pics'
    for folder in Path(workdir).glob('*'):
        for img in folder.glob('*'):
            if img.suffix in {'.jpg', '.png', '.jpeg'}:
                yield img


log.add('file_{time}.log')
log.info('Start')
prompts=('quality', 'noisiness', 'sharpness','natural','complexity', 
         ('plant photo','not plant photo'), 
         ('flower photo', 'not flower photo'),
         ('plant leaf photo', 'not plant leaf photo'))
# m = clip_iqa(prompts=prompts, model_name_or_path='openai/clip-vit-base-patch32')
m = clip_iqa(prompts=prompts,
             model_name_or_path='openai/clip-vit-large-patch14').to('cuda')
log.info('Model loaded')
log.info(f'Prompts:{prompts}')

out = open('result.txt', 'w')
out.write(str(prompts)+'\n')
for img_file in scan_files():
    # unsqueeze to 4-d tensor 
    # img = torch.randint(255,(2,3,224,224)).float()
    img = torchvision.io.read_image(img_file).unsqueeze(0).to('cuda')
    s = m(img)
    #s = m(torch.tensor(img))
    values = ','.join([f'{i.item():.2f}' for i in s.values()])
    log.info(f'{img_file.stem}, {values}')
    out.write(f'{img_file},{values}\n')
log.info('Done')

