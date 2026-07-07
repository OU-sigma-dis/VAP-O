import csv
import glob
import random
from pathlib import Path

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

from utilities.transforms import Augmentation

# --- 設定項目 ---
SPH_BASE_DIR = Path.home() / "Corpus/Switchboard/swb1_LDC97S62"
TRANS_BASE_DIR = Path.home() / "Corpus/Switchboard/swb_ms98_transcriptions"
OUTPUT_DIR = Path.home() / "data/switchboard/vap-o_dataset"

SAMPLE_RATE = 16000
FRAME_HZ = 20
SEED = 42

NON_SPEECH_MARKERS = [
    "[silence]",
    "[laughter]",
    "[noise]",
    "[vocalized-noise]",
]

# Augmentation設定
AUGMENTATION = Augmentation(
    probability=0.5,  # 0.5の確率でAugmentationを適用
    noise_amplitude=0.01,
    pitch_steps=[-2, -1, 1, 2],
    freq_mask_param=100,
    iid_masks=True,
    sample_rate=SAMPLE_RATE,
    device="cpu",
)


def convert_sph_to_wav(sph_path, output_path):
    """SPHファイルをWAVファイルに変換し、16kHzにリサンプリング"""
    try:
        audio, sr = librosa.load(sph_path, sr=None, mono=False)

        if len(audio.shape) > 1:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            audio_resampled = audio_resampled.T
            sf.write(output_path, audio_resampled, SAMPLE_RATE)
            return [output_path]
        else:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sf.write(output_path, audio_resampled, SAMPLE_RATE)
            return [output_path]
    except Exception as e:
        print(f"Error converting {sph_path}: {e}")
        return []


def extract_word_text(raw_word):
    """
    転記トークンからクリーンな単語テキストを抽出する。

    入力: 転記ファイルの単語フィールド
    出力: 発話単語の文字列。非発話トークンの場合はNone。

    処理パターン:
      - [laughter-WORD] -> WORD（笑いながらの発話）
      - [spoken/correct] -> correct（発音間違いの正しい形）
      - NON_SPEECH_MARKERSやタグ -> None
    """
    if raw_word in NON_SPEECH_MARKERS:
        return None
    if raw_word.startswith("<"):
        return None
    if raw_word.startswith("[laughter-") and raw_word.endswith("]"):
        return raw_word[len("[laughter-"):-1]
    if raw_word.startswith("[") and "/" in raw_word and raw_word.endswith("]"):
        return raw_word[raw_word.index("/") + 1:-1]
    return raw_word


def parse_word_file(word_path):
    """
    単語単位の転写ファイル（word.text）から発話単語とタイミングを抽出する。

    入力: word.textファイルのパス
    出力: [(start_time, end_time, word_text), ...] 形式のリスト（非発話トークンは除外）
    """
    words = []

    with open(word_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            start_time = float(parts[1])
            end_time = float(parts[2])
            raw_word = parts[3]
            word_text = extract_word_text(raw_word)
            if word_text is not None:
                words.append((start_time, end_time, word_text))

    return words


def merge_overlapping_segments(segments):
    """重複する発話区間をマージ"""
    if not segments:
        return []

    segments.sort(key=lambda x: x[0])
    merged = [segments[0]]

    for current in segments[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1][1] = max(last[1], current[1])
        else:
            merged.append(current)

    return merged


def generate_frame_labels(vad_a, vad_b, words_all, duration):
    """
    20Hzフレームレベルのラベルを生成する。

    入力:
      vad_a: 話者AのVADセグメント [[start, end], ...]
      vad_b: 話者BのVADセグメント [[start, end], ...]
      words_all: 両話者の単語リスト [(start_time, word_text), ...]（start_timeでソート済み）
      duration: 音声の長さ（秒）
    出力:
      dict with keys:
        next_word: list[str] 各フレームから次に始まる発話単語
        onset_proximity: (n_frames, 2) 各話者のonset proximity（horizon=3.0秒）
      ※ VAはウィンドウサイズ変更に対応するため.ptに保存せず、dataloaderで動的に計算する
    """
    n_frames = int(duration * FRAME_HZ)

    # --- VA: 各フレームのvoice activity ---
    va = torch.zeros(n_frames, 2)
    for start, end in vad_a:
        s = int(start * FRAME_HZ)
        e = min(int(end * FRAME_HZ), n_frames)
        va[s:e, 0] = 1.0
    for start, end in vad_b:
        s = int(start * FRAME_HZ)
        e = min(int(end * FRAME_HZ), n_frames)
        va[s:e, 1] = 1.0

    # --- Onset Proximity: 各話者の次のonsetまでの近さ ---
    # onset_proximity[t, speaker] = max(0, 1 - distance_to_next_onset(t) / horizon)
    # horizon=3.0秒。reverse scanで効率的に計算。
    ONSET_HORIZON = 3.0
    horizon_frames = int(ONSET_HORIZON * FRAME_HZ)
    onset_proximity = torch.zeros(n_frames, 2)

    for speaker in range(2):
        speech = va[:, speaker]
        # onset = 非発話→発話の立ち上がりフレーム
        padded = torch.cat([torch.zeros(1), speech])
        onsets = set(((padded[1:] == 1) & (padded[:-1] == 0)).nonzero(as_tuple=True)[0].tolist())

        # reverse scan: 末尾から走査して次のonsetまでの距離を記録
        dist = horizon_frames + 1  # 初期値: horizon外
        for t in range(n_frames - 1, -1, -1):
            if t in onsets:
                dist = 0
            onset_proximity[t, speaker] = max(0.0, 1.0 - dist / horizon_frames)
            dist += 1

    # --- next_word: 各フレームから次に始まる発話単語 ---
    # two-pointer走査（フレームと単語リストは共に時間順）
    next_word = [""] * n_frames
    word_idx = 0
    for t in range(n_frames):
        frame_time = t / FRAME_HZ
        # frame_timeより前に開始済みの単語をスキップ
        while word_idx < len(words_all) and words_all[word_idx][0] < frame_time:
            word_idx += 1
        if word_idx < len(words_all):
            next_word[t] = words_all[word_idx][1]

    return {
        "next_word": next_word,
        "onset_proximity": onset_proximity,
    }


def get_session_files():
    """セッションファイルのリストを取得"""
    sessions = []

    for sph_file in glob.glob(str(SPH_BASE_DIR / "*/data/*.sph")):
        sph_path = Path(sph_file)
        session_id = sph_path.stem
        sessions.append(session_id)

    return sorted(list(set(sessions)))


def split_sessions(sessions):
    """セッションをtrain/val/testに分割"""
    # test: sw02001 - sw02184
    test_sessions = [s for s in sessions if 2001 <= int(s[2:]) <= 2184]

    # train, val: sw02185 - sw04940
    train_val_sessions = [s for s in sessions if int(s[2:]) >= 2185]

    # train_val_sessionsを9:1に分割
    random.seed(SEED)
    random.shuffle(train_val_sessions)

    split_idx = int(len(train_val_sessions) * 0.9)
    train_sessions = train_val_sessions[:split_idx]
    val_sessions = train_val_sessions[split_idx:]

    return train_sessions, val_sessions, test_sessions


def process_session(session_id, is_train=False):
    """セッションを処理してCSV行を生成"""
    # SPHファイルを見つける
    sph_files = list(SPH_BASE_DIR.glob(f"*/data/{session_id}.sph"))
    if not sph_files:
        return None

    sph_path = sph_files[0]

    # 出力ファイルパス
    wav_output_dir = OUTPUT_DIR / "audio"
    wav_output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_output_dir / f"{session_id}.wav"

    label_output_dir = OUTPUT_DIR / "labels"
    label_output_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_output_dir / f"{session_id}.pt"

    # SPHをWAVに変換
    converted_files = convert_sph_to_wav(sph_path, wav_path)
    if not converted_files:
        return None

    # 転写ファイルのディレクトリ
    session_num = session_id[3:]  # sw02001 -> 2001
    trans_dir = TRANS_BASE_DIR / session_num[:2] / session_num

    # 話者A, Bの単語レベル転写ファイル
    word_a_path = trans_dir / f"sw{session_num}A-ms98-a-word.text"
    word_b_path = trans_dir / f"sw{session_num}B-ms98-a-word.text"

    if not word_a_path.exists() or not word_b_path.exists():
        return None

    # 単語リストを抽出（word.textから）
    words_a = parse_word_file(word_a_path)
    words_b = parse_word_file(word_b_path)

    # VAD区間を単語レベルのアノテーションから構築
    vad_a = merge_overlapping_segments([[s, e] for s, e, _ in words_a])
    vad_b = merge_overlapping_segments([[s, e] for s, e, _ in words_b])

    # trainセットの場合のみ、0.5の確率で話者を入れ替える
    if is_train and random.random() < 0.5:
        audio, _ = librosa.load(converted_files[0], sr=SAMPLE_RATE, mono=False)
        if len(audio.shape) > 1 and audio.shape[0] == 2:
            audio_swapped = audio[::-1]
            sf.write(wav_path, audio_swapped.T, SAMPLE_RATE)

        # VADラベルと単語リストを入れ替える
        vad_a, vad_b = vad_b, vad_a
        words_a, words_b = words_b, words_a

    # trainセットの場合のみ、Augmentationを適用（内部で0.5の確率制御）
    if is_train:
        audio, _ = librosa.load(converted_files[0], sr=SAMPLE_RATE, mono=False)

        if len(audio.shape) > 1 and audio.shape[0] == 2:
            audio_tensor = torch.from_numpy(audio).float()
            original_length = audio_tensor.shape[1]

            torch.use_deterministic_algorithms(False)

            augmented_channels = []
            for channel in audio_tensor:
                augmented_channel = AUGMENTATION(channel.unsqueeze(0))
                augmented_channel = augmented_channel.squeeze(0)

                if augmented_channel.shape[0] > original_length:
                    augmented_channel = augmented_channel[:original_length]
                elif augmented_channel.shape[0] < original_length:
                    padding = original_length - augmented_channel.shape[0]
                    augmented_channel = torch.cat(
                        [augmented_channel, torch.zeros(padding)]
                    )

                augmented_channels.append(augmented_channel)

            augmented_audio = torch.stack(augmented_channels, dim=0)

            torch.use_deterministic_algorithms(True)

            sf.write(wav_path, augmented_audio.numpy().T, SAMPLE_RATE)
        else:
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

            torch.use_deterministic_algorithms(False)
            augmented_audio = AUGMENTATION(audio_tensor)
            torch.use_deterministic_algorithms(True)

            sf.write(wav_path, augmented_audio.squeeze(0).numpy(), SAMPLE_RATE)

    # 音声の長さを取得
    audio, _ = librosa.load(converted_files[0], sr=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

    # 両話者の単語を統合し、開始時刻でソート（next_word用にstart_timeとword_textのみ）
    words_all = sorted(
        [(s, w) for s, _, w in words_a] + [(s, w) for s, _, w in words_b],
        key=lambda x: x[0],
    )

    # 20Hzフレームラベルを生成・保存
    labels = generate_frame_labels(vad_a, vad_b, words_all, duration)
    torch.save(labels, label_path)

    print(
        f"Processed {session_id}: duration={duration:.2f}s, "
        f"vad_a={len(vad_a)} segments, vad_b={len(vad_b)} segments, "
        f"frames={labels['va'].shape[0]}"
    )

    return {
        "audio_path": str(wav_path),
        "label_path": str(label_path),
        "start": 0,
        "end": duration,
        "session": session_id,
        "dataset": "switchboard",
        "vad_list": [vad_a, vad_b],
    }


def main():
    """メイン処理"""
    print("Switchboard VAP dataset creation started...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sessions = get_session_files()
    print(f"Found {len(sessions)} sessions")

    train_sessions, val_sessions, test_sessions = split_sessions(sessions)
    print(
        f"Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}"
    )

    for split_name, split_session_list in [
        ("train", train_sessions),
        ("val", val_sessions),
        ("test", test_sessions),
    ]:
        print(f"Processing {split_name} split...")

        csv_path = OUTPUT_DIR / f"{split_name}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["audio_path", "label_path", "start", "end", "session", "dataset", "vad_list"]
            )

            for session_id in tqdm(split_session_list, desc=f"Processing {split_name}"):
                result = process_session(session_id, is_train=(split_name == "train"))
                if result:
                    writer.writerow(
                        [
                            result["audio_path"],
                            result["label_path"],
                            result["start"],
                            result["end"],
                            result["session"],
                            result["dataset"],
                            str(result["vad_list"]),
                        ]
                    )
                else:
                    print(
                        f"[WARNING] Skipping session {session_id} due to processing error."
                    )

    print("Dataset creation completed!")


if __name__ == "__main__":
    main()
