import time

import torch
import torch.nn as nn

from discriminator import Discriminator
from generator import Generator
import data_loader


def train_model(G, D, dataloader, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    g_lr = 0.0001
    d_lr = 0.0004
    beta1 = 0.0
    beta2 = 0.9

    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    z_dim = 20
    mini_batch_size = 64

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True

    num_train_images = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []

    for epoch in range(num_epochs):
        t_epoch_start = time.time()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print("----------")
        print(f"Epoch {epoch}/{num_epochs}")
        print("----------")
        print(" train")

        for images in dataloader:
            if images.size()[0] == 1:
                continue

            images = images.to(device)

            mini_batch_size = images.size()[0]
            label_real = torch.full((mini_batch_size,), 1.0).to(device)
            label_fake = torch.full((mini_batch_size,), 0.0).to(device)

            # Dの学習
            d_out_real = D(images)

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # Gの学習
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake = D(fake_images)

            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # 記録
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print("-----------")
        print(
            f"epoch {epoch} || Epoch_D_Loss:{epoch_d_loss/batch_size:.4f} || Epoch_G_Loss:{epoch_g_loss/batch_size:.4f}"
        )
        print(f"timer: {t_epoch_finish-t_epoch_start:.4f}")
        t_epoch_start = time.time()
    return G, D


if __name__ == "__main__":

    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(z_dim=20, image_size=64)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    G.apply(weights_init)
    D.apply(weights_init)

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

    G_update, D_update = train_model(
        G, D, dataloader=train_dataloader, num_epochs=num_epochs
    )

    torch.save(
        {
            "G": G_update.state_dict(),
            "D": D_update.state_dict(),
        },
        "model.pth",
    )
