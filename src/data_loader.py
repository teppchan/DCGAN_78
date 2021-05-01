from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

def make_datapath_list():
    train_img_list = []

    for img_idx in range(200):
        img_path = f"data/img_78/img_7_{img_idx}.jpg"
        train_img_list.append(img_path)

        img_path = f"data/img_78/img_8_{img_idx}.jpg"
        train_img_list.append(img_path)
    
    return train_img_list


class ImageTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose(
            [                    
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ],
        )

    def __call__(self, img):
        return self.data_transform(img)

class GAN_Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img)

        return img_transformed


if __name__ == "__main__":
    train_img_list=make_datapath_list()

    mean = (0.5, )
    std = (0.5, )
    train_dataset = GAN_Img_Dataset(
        file_list=train_img_list,
        transform=ImageTransform(mean, std),
    )

    batch_size = 64

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    batch_iterator = iter(train_dataloader)
    imges = next(batch_iterator)
    print(imges.size())