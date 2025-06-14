import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os
import numpy as np

# ハイパーパラメータ
EPOCHS = 100 # GANの学習は時間がかかることがあります
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
LATENT_DIM = 100 # 生成器に入力するランダムノイズの次元
BETA1 = 0.5 # Adamオプティマイザの推奨パラメータ

# 画像のサイズとチャンネル数
IMG_SIZE = 28
CHANNELS = 1
IMG_SHAPE = (CHANNELS, IMG_SIZE, IMG_SIZE)

# 結果を保存するディレクトリ
if not os.path.exists('gan_results'):
    os.makedirs('gan_results')

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 2. データセットの準備
# 画像を-1から1の範囲に正規化する
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # 平均0.5, 標準偏差0.5で正規化
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# 3. Generator（生成器）の定義
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), # バッチ正規化で学習を安定させる
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(IMG_SHAPE))),
            nn.Tanh() # 出力を-1から1の範囲に
        )

    def forward(self, z):
        # z: (バッチサイズ, LATENT_DIM)
        img = self.model(z)
        # 出力を画像の形状に整形
        img = img.view(img.size(0), *IMG_SHAPE)
        return img


# 4. Discriminator（識別器）の定義
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(IMG_SHAPE)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid() # 画像が本物である確率(0-1)を出力
        )

    def forward(self, img):
        # img: (バッチサイズ, 1, 28, 28)
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# 損失関数
adversarial_loss = nn.BCELoss() # 二値交差エントロピー

# モデルのインスタンス化
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# オプティマイザ (GeneratorとDiscriminatorで別々に定義)
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))


# 生成画像を保存するための固定ノイズ
fixed_noise = torch.randn(64, LATENT_DIM).to(device)

for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(train_loader):

        # --- ラベルの準備 ---
        # 本物の画像のラベルは1
        valid = torch.ones(imgs.size(0), 1, requires_grad=False).to(device)
        # 偽物の画像のラベルは0
        fake = torch.zeros(imgs.size(0), 1, requires_grad=False).to(device)

        # 本物の画像をデバイスに移動
        real_imgs = imgs.to(device)

        #  Generatorの学習
        optimizer_G.zero_grad()

        # ランダムノイズを生成
        z = torch.randn(imgs.size(0), LATENT_DIM).to(device)

        # 偽画像を生成
        gen_imgs = generator(z)

        # Generatorの損失を計算 (Discriminatorを騙せたら損失が小さくなる)
        # Discriminatorが偽画像を「本物(valid)」と誤認識するほど良い
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        # 勾配を計算し、Generatorを更新
        g_loss.backward()
        optimizer_G.step()

        #  Discriminatorの学習
        optimizer_D.zero_grad()

        # 本物の画像に対する損失
        real_loss = adversarial_loss(discriminator(real_imgs), valid)

        # 偽の画像に対する損失 (Generatorの勾配は更新しないようにdetachする)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        # 本物と偽物の損失を合計してDiscriminatorの損失とする
        d_loss = (real_loss + fake_loss) / 2

        # 勾配を計算し、Discriminatorを更新
        d_loss.backward()
        optimizer_D.step()

        # --- 進捗の表示 ---
        if (i + 1) % 200 == 0:
            print(
                f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i+1}/{len(train_loader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )

    # --- エポックごとに生成画像を保存 ---
    generator.eval() # 生成時は評価モード
    with torch.no_grad():
        generated_images = generator(fixed_noise).detach().cpu()
        save_image(generated_images, f"gan_results/epoch_{epoch+1}.png", nrow=8, normalize=True)
    generator.train() # 学習モードに戻す

# 7. 画像の生成と可視化
print("\n--- Training finished. Showing generated images. ---")

# 学習済みのGeneratorから画像を生成
generator.eval()
with torch.no_grad():
    # 新しいノイズから画像を生成
    z = torch.randn(64, LATENT_DIM).to(device)
    final_images = generator(z).cpu()

# 生成された画像を表示
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Final Generated Images")
plt.imshow(make_grid(final_images, padding=2, normalize=True).permute(1, 2, 0))
plt.show()