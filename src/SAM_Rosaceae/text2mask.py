
from PIL import Image
from lang_sam import LangSAM
from pathlib import Path

img_file = Path(r'R:\test.jpg')
model = LangSAM('vit-l', "<path/to/checkpoint>")
img = Image.open(img_file).convert('RGB')
text_prompts = 'plant,flower,leaf,fruit,stem,root,background'.split(',')
masks, boxes, phrases, logits = model.predict(img, text_prompts[0])