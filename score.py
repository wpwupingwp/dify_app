from timeit import default_timer as timer
a = timer()
from torchmetrics.multimodal import CLIPImageQualityAssessment as clip_iqa
import torch
import torchvision

prompts=('quality', 'noisiness', 'sharpness','natural','complexity', 
         ('plant photo','not plant photo'), 
         ('flower photo', 'not flower photo'))
m = clip_iqa(prompts=prompts, model_name_or_path='openai/clip-vit-base-patch32')
print('load modules', timer()-a)
print(prompts)
a = timer()
for i in range(1,10):
    img_file = f'/tmp/a/test/{i}.jpg'
    img = torchvision.io.read_image(img_file).unsqueeze(0)
#    img = torch.randint(255,(2,3,224,224)).float()
    s = m(img)
    #s = m(torch.tensor(img))
    values = [f'{i.item():.2f}' for i in s.values()]
    print(img_file, values)
b = timer()
print('ave time', (b-a)/9, 'seconds per image')
