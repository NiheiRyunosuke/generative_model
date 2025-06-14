import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os

EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LATENT_DIM = 20  # 潜在変数の次元
if not os.path.exists('vae_results'):
    os.makedirs('vae_results')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# データローダーを作成
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. VAEモデルの定義
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # --- エンコーダ ---
        # 入力層 (28*28=784) -> 隠れ層 (400)
        self.fc1 = nn.Linear(784, 400)
        # 隠れ層 -> 潜在空間の平均 (mu)
        self.fc21 = nn.Linear(400, LATENT_DIM)
        # 隠れ層 -> 潜在空間の対数分散 (logvar)
        self.fc22 = nn.Linear(400, LATENT_DIM)

        # --- デコーダ ---
        # 潜在空間 -> 隠れ層
        self.fc3 = nn.Linear(LATENT_DIM, 400)
        # 隠れ層 -> 出力層 (784)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        """エンコーダ: 入力xから潜在空間のパラメータ(mu, logvar)を出力"""
        h = F.relu(self.fc1(x))
        mu = self.fc21(h)
        logvar = self.fc22(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """再パラメータ化トリック: muとlogvarから潜在変数zをサンプリング"""
        std = torch.exp(0.5 * logvar)  # 標準偏差を計算
        eps = torch.randn_like(std)    # 標準正規分布からノイズをサンプリング
        return mu + eps * std          # z = mu + epsilon * std

    def decode(self, z):
        """デコーダ: 潜在変数zから画像を復元"""
        h = F.relu(self.fc3(z))
        # Sigmoid関数でピクセル値を0〜1の範囲に変換
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        """モデルの順伝播"""
        # xをフラットなベクトルに変形
        x_flat = x.view(-1, 784)
        # エンコード
        mu, logvar = self.encode(x_flat)
        # 潜在変数をサンプリング
        z = self.reparameterize(mu, logvar)
        # デコード
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# 4. 損失関数とオプティマイザ
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def loss_function(recon_x, x, mu, logvar):
    """損失関数を計算"""
    # 1. 再構築誤差 (Reconstruction Loss)
    # 二値交差エントロピーを使用。元の画像と復元画像のピクセルごとの差を測る。
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # 2. KLダイバージェンス (Kullback-Leibler Divergence)
    # エンコーダが出力した分布と標準正規分布との近さを測る正則化項。
    # D_KL(N(mu, sigma) || N(0, 1))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# 5. 学習と評価のループ
def train(epoch):
    model.train()  # モデルを学習モードに
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.4f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(epoch):
    model.eval()  # モデルを評価モードに
    test_loss = 0
    with torch.no_grad(): # 勾配計算を無効化
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                # 最初のバッチの元画像と再構築画像を比較して保存
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), f'results/reconstruction_{epoch}.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

# --- 学習の実行 ---
for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test(epoch)
    # エポックごとにランダムな潜在変数から画像を生成して保存
    with torch.no_grad():
        sample = torch.randn(64, LATENT_DIM).to(device)
        generated_images = model.decode(sample).cpu()
        save_image(generated_images.view(64, 1, 28, 28), f'results/sample_{epoch}.png')


# 6. 画像の生成と可視化
print("\n--- Generating new images from random latent variables ---")

# 学習済みモデルを使って64枚の画像を生成
with torch.no_grad():
    # 潜在空間からランダムにベクトルをサンプリング
    sample = torch.randn(64, LATENT_DIM).to(device)
    # デコーダで画像を生成
    generated_images = model.decode(sample).cpu()

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(make_grid(generated_images.view(64, 1, 28, 28), padding=2, normalize=True).permute(1, 2, 0))
plt.show()

print("\n--- Showing reconstructed images ---")
# テストデータから1バッチ取得
original_images, _ = next(iter(test_loader))
original_images = original_images.to(device)

# 画像を再構築
reconstructed_images, _, _ = model(original_images)
reconstructed_images = reconstructed_images.cpu().view_as(original_images)

# 上段に元画像、下段に再構築画像を表示
fig, axes = plt.subplots(2, 8, figsize=(12, 4))
fig.suptitle("Top: Original Images, Bottom: Reconstructed Images")
for i in range(8):
    axes[0, i].imshow(original_images[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructed_images[i].detach().squeeze(), cmap='gray')
    axes[1, i].axis('off')
plt.show()