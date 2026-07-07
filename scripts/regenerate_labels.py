"""既存の.ptラベルファイルに onset_proximity を追加し、s_05/s_10/s_20 を削除するスクリプト。

既存の audio, word_tokens, next_filler_class はそのまま保持する。
onset_proximity は CSV の vad_list から再計算する。

使い方:
  python scripts/regenerate_labels.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

FRAME_HZ = 20
ONSET_HORIZON = 3.0


def compute_onset_proximity(vad_list, n_frames):
    """vad_list から onset proximity を計算する。

    入力:
      vad_list: [[[s, e], ...], [[s, e], ...]] 2話者分のVADセグメント
      n_frames: フレーム数
    出力: (n_frames, 2) の onset proximity テンソル
    """
    horizon_frames = int(ONSET_HORIZON * FRAME_HZ)
    onset_proximity = torch.zeros(n_frames, 2)

    # VA を構築
    va = torch.zeros(n_frames, 2)
    for speaker, segments in enumerate(vad_list):
        for start, end in segments:
            s = int(start * FRAME_HZ)
            e = min(int(end * FRAME_HZ), n_frames)
            va[s:e, speaker] = 1.0

    for speaker in range(2):
        speech = va[:, speaker]
        # onset = 非発話→発話の立ち上がりフレーム
        padded = torch.cat([torch.zeros(1), speech])
        onsets = set(
            ((padded[1:] == 1) & (padded[:-1] == 0)).nonzero(as_tuple=True)[0].tolist()
        )

        # reverse scan
        dist = horizon_frames + 1
        for t in range(n_frames - 1, -1, -1):
            if t in onsets:
                dist = 0
            onset_proximity[t, speaker] = max(0.0, 1.0 - dist / horizon_frames)
            dist += 1

    return onset_proximity


def main():
    # プロジェクトルートに移動
    import os
    os.chdir(Path(__file__).resolve().parent.parent)

    csv_paths = [
        "/Users/onishi/data/switchboard/vap-o_dataset/train.csv",
        "/Users/onishi/data/switchboard/vap-o_dataset/val.csv",
        "/Users/onishi/data/switchboard/vap-o_dataset/test.csv",
    ]

    # 絶対パスに変換
    csv_paths = [str(Path(p).resolve()) for p in csv_paths]

    total_updated = 0

    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            print(f"CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, converters={"vad_list": json.loads})
        split_name = Path(csv_path).stem
        print(f"\n=== {split_name}: {len(df)} sessions ===")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            label_path = row["label_path"]
            if not Path(label_path).exists():
                print(f"  Label not found: {label_path}")
                continue

            # 既存ラベルをロード
            data = torch.load(label_path, map_location="cpu", weights_only=True)

            # フレーム数を既存データから取得
            n_frames = data["next_filler_class"].shape[0]

            # onset proximity を計算
            vad_list = row["vad_list"]
            onset_prox = compute_onset_proximity(vad_list, n_frames)

            # 新しいラベルを構築（s_05/s_10/s_20 を除去、onset_proximity を追加）
            new_data = {
                "word_tokens": data["word_tokens"],
                "next_filler_class": data["next_filler_class"],
                "onset_proximity": onset_prox,
            }
            # 互換性のため残す
            if "next_word" in data:
                new_data["next_word"] = data["next_word"]
            if "next_word_token" in data:
                new_data["next_word_token"] = data["next_word_token"]

            torch.save(new_data, label_path)
            total_updated += 1

    print(f"\n完了: {total_updated} ファイルを更新しました")


if __name__ == "__main__":
    main()
