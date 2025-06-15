import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import glob
import librosa
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt

# 学習設定
EPOCHS = 1000 # 本格的な学習には2500以上推奨
BATCH_SIZE = 16
LEARNING_RATE = 0.0002
ADAM_B1 = 0.8
ADAM_B2 = 0.99
LAMBDA_MEL = 45    # メルスペクトログラム損失の重み
LAMBDA_FM = 2      # Feature Matching損失の重み

# データセットと音声設定
DATA_PATH = "./LJSpeech-1.1/wavs"
SAMPLE_RATE = 22050 # LJSpeechのサンプルレート
SEGMENT_SIZE = 8192 # 学習に使う音声の断片長

N_FFT = 1024
NUM_MELS = 80
HOP_SIZE = 256
WIN_SIZE = 1024
FMIN = 0
FMAX = 8000

# デバイスとディレクトリ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs("results_hifigan", exist_ok=True)

# 1. データセットの準備
def get_mel_spectrogram(y):
    # Librosaを使ってメルスペクトログラムを計算
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX)
    mel_basis = torch.from_numpy(mel_basis).float()
    
    stft = torch.stft(y, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=WIN_SIZE, window=torch.hann_window(WIN_SIZE), return_complex=True)
    spec = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-9)
    
    mel_spec = torch.matmul(mel_basis, spec)
    log_mel_spec = torch.log(mel_spec)
    return log_mel_spec

class AudioDataset(Dataset):
    def __init__(self, audio_files, segment_size):
        self.audio_files = audio_files
        self.segment_size = segment_size

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        filename = self.audio_files[idx]
        audio, sr = torchaudio.load(filename)
        audio = audio.squeeze(0)
        
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            audio = resampler(audio)

        # ランダムなセグメントを切り出す
        if audio.size(0) >= self.segment_size:
            start = np.random.randint(0, audio.size(0) - self.segment_size + 1)
            audio = audio[start:start + self.segment_size]
        else:
            audio = F.pad(audio, (0, self.segment_size - audio.size(0)))

        mel = get_mel_spectrogram(audio)
        return mel, audio

# 2. HiFi-GAN モデルの定義
# Generator
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=d * (kernel_size - 1) // 2))
            for d in dilation])
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2))
            for _ in dilation])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class Generator(nn.Module):
    def __init__(self, upsample_rates=(8, 8, 2, 2), upsample_kernel_sizes=(16, 16, 4, 4), upsample_initial_channel=512, resblock_kernel_sizes=(3, 7, 11)):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = nn.utils.weight_norm(nn.Conv1d(NUM_MELS, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, [(1, 3, 5), (1, 3, 5), (1, 3, 5)])):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

# Discriminators
class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super(ScaleDiscriminator, self).__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            nn.utils.weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator() for _ in range(3)
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# 3. 損失関数の定義
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
    return loss

def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        loss += l
    return loss

# 4. 学習の実行
if __name__ == '__main__':
    # データローダー
    audio_files = glob.glob(os.path.join(DATA_PATH, "*.wav"))
    dataset = AudioDataset(audio_files, SEGMENT_SIZE)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # モデル
    generator = Generator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # オプティマイザ
    optim_g = optim.AdamW(generator.parameters(), LEARNING_RATE, betas=[ADAM_B1, ADAM_B2])
    optim_d = optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), LEARNING_RATE, betas=[ADAM_B1, ADAM_B2])

    print("Starting Training...")
    for epoch in range(1, EPOCHS + 1):
        generator.train()
        mpd.train()
        msd.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for i, (mel, y) in enumerate(progress_bar):
            mel = mel.to(device)
            y = y.unsqueeze(1).to(device)

            # Generator
            y_g_hat = generator(mel)
            y_g_hat_mel = get_mel_spectrogram(y_g_hat.squeeze(1))

            # Discriminator
            optim_d.zero_grad()
            # MPD
            y_df_r, y_df_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f = discriminator_loss(y_df_r, y_df_g)
            # MSD
            y_ds_r, y_ds_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s = discriminator_loss(y_ds_r, y_ds_g)
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()
            
            # Generator
            optim_g.zero_grad()
            loss_mel = F.l1_loss(y_g_hat_mel, mel) * LAMBDA_MEL
            
            _, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            _, y_ds_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f = generator_loss(y_df_g)
            loss_gen_s = generator_loss(y_ds_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            loss_gen_all.backward()
            optim_g.step()

            progress_bar.set_postfix(loss_g=loss_gen_all.item(), loss_d=loss_disc_all.item())

        # 10エポックごとにサンプル音声を保存
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                # テスト用のデータを1つ取得
                test_mel, _ = dataset[0]
                test_mel = test_mel.unsqueeze(0).to(device)
                
                generated_audio = generator(test_mel)
                generated_audio = generated_audio.squeeze().cpu()
                
                torchaudio.save(f"results_hifigan/generated_epoch_{epoch}.wav", generated_audio.unsqueeze(0), SAMPLE_RATE)

                # スペクトログラムも保存
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                ax1.imshow(test_mel.squeeze(0).cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                ax1.set_title("Original Mel-Spectrogram")
                
                generated_mel = get_mel_spectrogram(generated_audio)
                ax2.imshow(generated_mel.numpy(), aspect='auto', origin='lower', cmap='viridis')
                ax2.set_title("Generated Mel-Spectrogram")
                plt.savefig(f"results_hifigan/spectrogram_comparison_epoch_{epoch}.png")
                plt.close()


    print("Training finished.")