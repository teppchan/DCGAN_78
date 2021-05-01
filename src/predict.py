import torch
import matplotlib.pyplot as plt

from generator import Generator
import data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


G = Generator(z_dim=20, image_size=64)
G.to(device)

checkpoint = torch.load("model.pth")
print(checkpoint.keys())
G.load_state_dict(checkpoint["G"])

batch_size = 8
z_dim = 20
fixed_z = torch.rand(batch_size, z_dim)
fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)
fixed_z = fixed_z.to(device)

fake_images = G(fixed_z)

train_img_list = data_loader.make_datapath_list()

mean = (0.5,)
std = (0.5,)
train_dataset = data_loader.GAN_Img_Dataset(
    file_list=train_img_list,
    transform=data_loader.ImageTransform(mean, std),
)

num_epochs = 200
batch_size = 64

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

batch_iterator = iter(train_dataloader)
images = next(batch_iterator)

fig = plt.figure(figsize=(15, 6))
for i in range(0, 5):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i][0].cpu().detach().numpy(), "gray")

    plt.subplot(2, 5, 5+i+1)
    plt.imshow(fake_images[i][0].cpu().detach().numpy(), "gray")

plt.savefig("G.png")