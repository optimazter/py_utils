import os
import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import SimpleITK as sitk
from rich.progress import Progress
import numpy as np



FLAIR_TIME_1 = 'flair_time01_on_middle_space'
FLAIR_TIME_2 = 'flair_time02_on_middle_space'
GROUND_TRUTH = 'ground_truth'


def create_nifti_dataset(load_dir: str, save_path: str, img_name: str, label_name: str, resize = Resize((256, 256))) -> None:
    labels = []
    imgs = []
    with Progress() as progress:
        patient_ids = os.listdir(load_dir)
        dataset_task = progress.add_task('Creating nifti dataset', total = len(patient_ids))
        for patient_id in patient_ids:
            nifti_img_path = f'{load_dir}/{patient_id}/{img_name}.nii'
            nifti_label_path = f'{load_dir}/{patient_id}/{label_name}.nii'
            if os.path.isfile(nifti_img_path) and os.path.isfile(nifti_label_path):
                
                nifti_imgs = sitk.ReadImage(nifti_img_path)
                nifti_labels = sitk.ReadImage(nifti_label_path)
                nifti_imgs_array = sitk.GetArrayFromImage(nifti_imgs)
                nifti_labels_array = sitk.GetArrayFromImage(nifti_labels)

                for i in range(nifti_imgs_array.shape[0]):
                    img = resize(torch.tensor(nifti_imgs_array[i], dtype = torch.float32).unsqueeze(0))
                    label = resize(torch.tensor(nifti_labels_array[i], dtype = torch.long).unsqueeze(0))
                    imgs.append(img)
                    labels.append(label)
                
                progress.update(dataset_task, advance=1)

        labels_t = torch.stack(labels, dim = 0)
        imgs_t = torch.stack(imgs, dim = 0)
        dataset = TensorDataset(imgs_t, labels_t)
        torch.save(dataset, save_path)
        print(f'Conversion completed, file saved at {save_path}')



def plot_nifti(nifti_imgs: list, masks: list = None, title: str = None):
    n_cols = len(nifti_imgs) // 2 if len(nifti_imgs) >= 4 else len(nifti_imgs)
    n_rows = len(nifti_imgs) // n_cols
    fig, axes = plt.subplots(nrows = n_rows, ncols = n_cols, figsize=(n_cols / n_rows * 8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(nifti_imgs):
            img = nifti_imgs[i].permute(1, 2, 0)
            if masks:
                mask = masks[i].permute(1, 2, 0)
                np.concatenate((img, mask, np.zeros_like(mask)), axis = 1)

            ax.imshow(img)
            ax.axis('off')
    if title:
        fig.suptitle(title, fontsize = 30)
    plt.show()

