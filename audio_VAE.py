import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
import os
import matplotlib.pyplot as plt

# ハイパーパラメータ
EPOCHS = 150 # データが複雑になったのでエポック数を増やす
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LATENT_DIM = 64  # 潜在変数の次元も少し増やす

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
AUDIO_LEN_SECONDS = 1 # SPEECHCOMMANDSデータセットは主に1秒の音声
AUDIO_LEN_SAMPLES = SAMPLE_RATE * AUDIO_LEN_SECONDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs("audioVAE_results", exist_ok=True)


# SPEECHCOMMANDSデータセットをダウンロード
try:
    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=".", download=True, subset="training")
except Exception as e:
    print("\n---")
    print("SPEECHCOMMANDSのダウンロードに失敗しました。ネットワーク接続を確認するか、手動でダウンロードしてください。")
    print(f"エラー詳細: {e}")
    print("---")
    exit()

# メルスペクトログラム変換
mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
).to(device)

# dBスケールへの変換
amp_to_db_transform = T.AmplitudeToDB().to(device)


def collate_fn(batch):
    """
    DataLoader用のカスタム関数。
    バッチ内の音声データの長さを揃える。
    """
    waveforms = []
    for item in batch:
        waveform, sr, label, speaker_id, utterance_number = item
        
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if waveform.shape[1] < AUDIO_LEN_SAMPLES:
            pad_len = AUDIO_LEN_SAMPLES - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :AUDIO_LEN_SAMPLES]
        
        waveforms.append(waveform)
        
    # バッチ内の波形を一つのテンソルにまとめる
    waveforms_tensor = torch.stack(waveforms, dim=0) # catからstackに変更
    
    return waveforms_tensor


# DataLoaderの作成
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4, # データ読み込みを高速化
    pin_memory=True # 高速化
)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        # --- エンコーダ ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # --- デコーダ ---
        self.decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(5,4), stride=(2,2), padding=(1,1), output_padding=(1,0)),
        )
        
        # ★★★ 修正点1: self.final_resize の行を削除 ★★★
        # self.final_resize = T.Resize([N_MELS, AUDIO_LEN_SAMPLES // HOP_LENGTH + 1]) # この行を削除

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 128, 8, 8)
        recon_spec = self.decoder(h)
        
        # ★★★ 修正点2: F.interpolate を使ってリサイズ ★★★
        target_size = [N_MELS, AUDIO_LEN_SAMPLES // HOP_LENGTH + 1]
        # 'bilinear'モードは4Dテンソル(Batch, Channel, Height, Width)を想定
        return F.interpolate(recon_spec, size=target_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


# 3. 学習の実行
if __name__ == '__main__':
    model = ConvVAE(LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training on SPEECHCOMMANDS dataset...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch_idx, waveform_batch in enumerate(train_loader):
            waveform_batch = waveform_batch.to(device)
            mel_spec = mel_spectrogram_transform(waveform_batch)
            data = amp_to_db_transform(mel_spec)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        avg_loss = train_loss / len(train_loader.dataset)
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
        
        # 10エポックごとに再構築したスペクトログラムを保存
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_data = next(iter(train_loader)).to(device)
                recon, _, _ = model(sample_data)
                
                # 比較画像を保存
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))
                fig.suptitle(f"Epoch {epoch} - Original vs Reconstructed")
                for i in range(min(5, BATCH_SIZE)):
                    axes[0, i].imshow(sample_data[i, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                    axes[0, i].set_title(f"Original {i}")
                    axes[0, i].axis('off')
                    axes[1, i].imshow(recon[i, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                    axes[1, i].set_title(f"Recon {i}")
                    axes[1, i].axis('off')

                plt.savefig(f"results_speech/reconstruction_epoch_{epoch}.png")
                plt.close()

    print("Training finished.")

    # 4. 新しい音声の生成と保存
    print("Generating new audio...")
    model.eval()

    # 音声波形に逆変換するための準備
    griffin_lim_transform = T.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH).to(device)
    db_to_amp = T.DBToAmplitude(ref=1.0, power=0.5).to(device)

    with torch.no_grad():
        z = torch.randn(5, LATENT_DIM).to(device)
        generated_spec_db = model.decode(z)

        # 可視化
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle("Generated Spectrograms")
        for i in range(5):
            axes[i].imshow(generated_spec_db[i, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
            axes[i].set_title(f"Generated {i}")
            axes[i].axis('off')
        plt.savefig("results_speech/generated_spectrograms.png")
        plt.close()
        
        # 音声波形に変換
        generated_spec_amp = db_to_amp(generated_spec_db)
        generated_waveform = griffin_lim_transform(generated_spec_amp)

        for i in range(generated_waveform.shape[0]):
            output_path = f"results_speech/generated_speech_{i}.wav"
            torchaudio.save(output_path, generated_waveform[i].cpu(), SAMPLE_RATE)

    print("Generated audio files are saved in 'results_speech' folder.")