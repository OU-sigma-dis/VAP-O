import random

import torch
import torchaudio.functional as AF
import torchaudio.transforms as AT


class Augmentation(torch.nn.Module):
    def __init__(
        self,
        probability: float = 0.5,
        noise_amplitude: float = 0.01,
        pitch_steps: list[int] = [-2, -1, 1, 2],
        freq_mask_param: int = 100,
        iid_masks: bool = True,
        sample_rate: int = 16_000,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.probability = probability
        self.sample_rate = sample_rate
        self.pitch_steps = pitch_steps
        self.noise_amplitude = noise_amplitude
        self.freq_mask_param = freq_mask_param
        self.iid_masks = iid_masks

        self.shift_pitch = PitchShift(
            pitch_steps=self.pitch_steps, sample_rate=sample_rate
        )
        self.frequency_masking = WaveformFrequencyMasking(
            freq_mask_param=freq_mask_param,
            iid_masks=iid_masks,
            sample_rate=sample_rate,
        )
        self.noise = AddGaussianNoise(max_amplitude=noise_amplitude)

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += f"\tnoise_amplitude={self.noise_amplitude},\n"
        s += f"\tpitch_steps={self.pitch_steps},\n"
        s += f"\tfreq_mask_param={self.freq_mask_param},\n"
        s += f"\tiid_masks={self.iid_masks},\n"
        s += f"\tsample_rate={self.sample_rate},\n"
        s += ")\n"
        return s

    def apply_all(self, x):
        x = self.shift_pitch(x)
        x = self.frequency_masking(x)
        return self.noise(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > self.probability:
            return x

        r = torch.rand(1)
        if r < 0.25:
            x = self.shift_pitch(x)
        elif 0.25 < r < 0.50:
            x = self.noise(x)
        elif 0.5 < r < 0.75:
            x = self.frequency_masking(x)
        else:
            x = self.apply_all(x)
        return x


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, max_amplitude=0.01):
        """
        :param min_amplitude: Minimum noise amplification factor
        :param max_amplitude: Maximum noise amplification factor
        :param p:
        """
        super().__init__()
        assert max_amplitude > 0.0
        self.max_amplitude = max_amplitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        noise -= noise.min()
        noise = 2 * self.max_amplitude * noise / noise.max()
        noise -= noise.max() / 2
        return x + noise


class PitchShift(torch.nn.Module):
    def __init__(
        self, pitch_steps: list[int] = [-2, -1, 1, 2], sample_rate: int = 16_000
    ):
        super().__init__()
        self.pitch_steps = pitch_steps
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cuda don't allow for use_deterministic_algorithms(True) so we untoggle it here"""
        p = random.choice(self.pitch_steps)
        torch.use_deterministic_algorithms(False)
        x = AF.pitch_shift(x, sample_rate=self.sample_rate, n_steps=p)
        torch.use_deterministic_algorithms(True)
        return x


class WaveformFrequencyMasking(torch.nn.Module):
    def __init__(
        self,
        window_time: float = 0.05,
        hop_time: float = 0.02,
        freq_mask_param: int = 100,
        iid_masks: bool = True,
        sample_rate: int = 16_000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = int(window_time * sample_rate)
        self.hop_length = int(hop_time * sample_rate)
        self.freq_mask_param = freq_mask_param
        self.iid_masks = iid_masks

        self.to_spectrogram = AT.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None
        )
        self.to_waveform = AT.InverseSpectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        self.frequency_masking = AT.FrequencyMasking(freq_mask_param, iid_masks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep everything on CPU
        spec = self.to_spectrogram(x)  # (B, C, Freq, Time)
        spec_fm = self.frequency_masking(spec.real)
        spec.real = spec_fm
        return self.to_waveform(spec)


def test():
    print("Testing Audio Transforms...")
    print("=" * 60)

    import torchaudio

    device = "cpu"
    print(f"Using device: {device}")

    # Load sample audio
    sample_path = "sample_audio/sample.wav"
    try:
        waveform, sample_rate = torchaudio.load(sample_path)
        print(f"Loaded audio: {sample_path}")
        print(f"Original shape: {waveform.shape}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {waveform.shape[1] / sample_rate:.2f} seconds")
    except Exception as e:
        print(f"Error loading audio: {e}")
        # Create dummy audio if file doesn't exist
        sample_rate = 16000
        duration = 2.0
        waveform = torch.randn(1, int(sample_rate * duration))
        print(f"Using dummy audio: shape {waveform.shape}")

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print(f"Converted to mono: {waveform.shape}")

    # Resample if needed
    if sample_rate != 16000:
        import torchaudio.functional as F

        waveform = F.resample(waveform, sample_rate, 16000)
        sample_rate = 16000
        print(f"Resampled to 16kHz: {waveform.shape}")

    waveform = waveform.to(device)
    print()

    # Test individual transforms
    print("Testing Individual Transforms:")
    print("-" * 40)

    # Test AddGaussianNoise
    print("1. AddGaussianNoise")
    noise_transform = AddGaussianNoise(max_amplitude=0.01).to(device)
    try:
        noisy_audio = noise_transform(waveform.clone())
        print(f"   Input range: [{waveform.min():.6f}, {waveform.max():.6f}]")
        print(f"   Output range: [{noisy_audio.min():.6f}, {noisy_audio.max():.6f}]")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()

    # Test PitchShift
    print("2. PitchShift")
    pitch_transform = PitchShift(
        pitch_steps=[-2, -1, 1, 2], sample_rate=sample_rate
    ).to(device)
    try:
        torch.use_deterministic_algorithms(False)  # Required for pitch shift
        pitched_audio = pitch_transform(waveform.clone())
        print(f"   Input shape: {waveform.shape}")
        print(f"   Output shape: {pitched_audio.shape}")
        print("   ✓ Success")
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()

    # Test WaveformFrequencyMasking
    print("3. WaveformFrequencyMasking")
    freq_mask_transform = WaveformFrequencyMasking(
        freq_mask_param=100, iid_masks=True, sample_rate=sample_rate
    ).to(device)
    try:
        masked_audio = freq_mask_transform(waveform.clone())
        print(f"   Input shape: {waveform.shape}")
        print(f"   Output shape: {masked_audio.shape}")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()

    # Test Full Augmentation Pipeline
    print("Testing Full Augmentation Pipeline:")
    print("-" * 40)

    augmentation = Augmentation(
        probability=1.0,  # Always apply for testing
        noise_amplitude=0.01,
        pitch_steps=[-2, -1, 1, 2],
        freq_mask_param=100,
        iid_masks=True,
        sample_rate=sample_rate,
    ).to(device)

    print("Augmentation config:")
    print(augmentation)

    try:
        # Test multiple applications
        for i in range(3):
            torch.use_deterministic_algorithms(False)
            augmented_audio = augmentation(waveform.clone())
            torch.use_deterministic_algorithms(True)
            print(
                f"   Run {i + 1}: Input {waveform.shape} -> Output {augmented_audio.shape}"
            )
        print("   ✓ Full pipeline success")
    except Exception as e:
        print(f"   ✗ Pipeline error: {e}")
        import traceback

        traceback.print_exc()

    # Save audio samples to output_debug/
    print("Saving audio samples to output_debug/:")
    print("-" * 40)

    import os

    output_dir = "output_debug"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Save original audio
        original_path = os.path.join(output_dir, "original.wav")
        torchaudio.save(original_path, waveform.cpu(), sample_rate)
        print(f"   Original audio saved: {original_path}")

        # Save noisy audio
        noisy_audio = AddGaussianNoise(max_amplitude=0.01)(waveform.clone())
        noisy_path = os.path.join(output_dir, "noisy.wav")
        torchaudio.save(noisy_path, noisy_audio.cpu(), sample_rate)
        print(f"   Noisy audio saved: {noisy_path}")

        # Save pitch shifted audio
        torch.use_deterministic_algorithms(False)
        pitch_transform = PitchShift(
            pitch_steps=[2], sample_rate=sample_rate
        )  # Fixed pitch step
        pitched_audio = pitch_transform(waveform.clone())
        pitched_path = os.path.join(output_dir, "pitched.wav")
        torchaudio.save(pitched_path, pitched_audio.cpu(), sample_rate)
        print(f"   Pitch shifted audio saved: {pitched_path}")
        torch.use_deterministic_algorithms(True)

        # Save frequency masked audio
        freq_mask_transform = WaveformFrequencyMasking(
            freq_mask_param=100, iid_masks=True, sample_rate=sample_rate
        )
        masked_audio = freq_mask_transform(waveform.clone())
        masked_path = os.path.join(output_dir, "freq_masked.wav")
        torchaudio.save(masked_path, masked_audio.cpu(), sample_rate)
        print(f"   Frequency masked audio saved: {masked_path}")

        # Save fully augmented audio
        augmentation_cpu = Augmentation(
            probability=1.0,
            noise_amplitude=0.01,
            pitch_steps=[-1, 1],  # Smaller range for stability
            freq_mask_param=50,  # Smaller mask for better audio quality
            iid_masks=True,
            sample_rate=sample_rate,
        )

        torch.use_deterministic_algorithms(False)
        for i in range(3):
            augmented_audio = augmentation_cpu(waveform.clone())
            augmented_path = os.path.join(output_dir, f"augmented_{i + 1}.wav")
            torchaudio.save(augmented_path, augmented_audio.cpu(), sample_rate)
            print(f"   Augmented audio {i + 1} saved: {augmented_path}")
        torch.use_deterministic_algorithms(True)

        print(f"   ✓ All audio samples saved to {output_dir}/")

    except Exception as e:
        print(f"   ✗ Error saving audio: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 60)
    print("Transform testing completed!")


if __name__ == "__main__":
    test()
