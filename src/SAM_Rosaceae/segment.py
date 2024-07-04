from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from loguru import logger as log

from preprocess import loader

sam_checkpoint = Path(r'F:\model\sam_vit_l_0b3195.pth')
model_type = 'vit_l'
device = 'cpu'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    # official parameters
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100, )


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def save_mask(masks: list, out_file: Path, info_file: Path
              ) -> tuple[Path, Path]:
    metadata = list()
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        cv2.imwrite(str(out_file), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]], ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    with open(info_file, 'a') as f:
        f.write('\n'.join(metadata))
    return out_file, info_file


def get_masks(image_file: Path) -> list:
    image = cv2.imread(str(image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    log.info(f'Generated {len(masks)} from {image_file}')

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    x = input('pause')
    plt.close()
    return masks


def main(input_folder: Path):
    info_file = input_folder / 'metainfo.csv'
    header = ('id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,'
              'point_input_y,predicted_iou,stability_score,crop_box_x0,'
              'crop_box_y0,crop_box_w,crop_box_h')
    info_file.write_text(header)
    for image_file in loader(input_folder):
        masks = get_masks(image_file)
        mask_file = image_file.with_name(image_file.stem+'-mask.png')

        save_mask(masks, mask_file, info_file)


if __name__ == '__main__':
    input_folder = Path(r'R:\test')
    main(input_folder)