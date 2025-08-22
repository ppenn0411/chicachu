# -*- coding: utf-8 -*-
"""
프레임 피처(.npy) → 시퀀스 윈도우 → 멀티태스크 라벨(surface/horizontal/vertical)
- 윈도우: seq_len, stride
- EMA / Kalman(옵션) / Δ feature(옵션)
- LOSO를 위한 group(person) / clip ID 저장
- N/A 라벨에 대한 마스크 저장

입력:
  data_<tag>/<라벨>/<사람>/*.npy   ← --data_root 생략 시 자동으로 이 경로 사용

출력(버전별 분리):
  artifacts/<tag>/
    X_data_mt_<tag>.npy
    y_surface_mt_<tag>.npy
    y_horizontal_mt_<tag>.npy
    y_vertical_mt_<tag>.npy
    mask_surface_mt_<tag>.npy
    mask_horizontal_mt_<tag>.npy
    mask_vertical_mt_<tag>.npy
    groups_mt_<tag>.npy
    clips_mt_<tag>.npy
    encoder_meta_<tag>.json
    manifest.json
"""

import os
import glob
import json
import argparse
import numpy as np
from typing import Dict, List

os.environ["PYTHONUTF8"] = "1"

# -------------------- 라벨 정의 --------------------
KOREAN_LABELS = [
    '왼쪽-협측','중앙-협측','오른쪽-협측',
    '왼쪽-구개측','중앙-구개측','오른쪽-구개측',
    '왼쪽-설측','중앙-설측','오른쪽-설측',
    '오른쪽-위-씹는면','왼쪽-위-씹는면','왼쪽-아래-씹는면','오른쪽-아래-씹는면'
]

SURFACES = ['Outside', 'Inside', 'Occlusal']
HORIZONTALS = ['Left', 'Front', 'Right']
VERTICALS = ['Lower', 'Upper']  # Inside/Occlusal이 아닌 경우는 N/A 처리

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# -------------------- 유틸 --------------------
def label_to_coarse(lbl: str) -> Dict[str, str]:
    # surface
    if '씹는면' in lbl:
        surface = 'Occlusal'
    elif ('구개측' in lbl) or ('설측' in lbl):
        surface = 'Inside'
    else:
        surface = 'Outside'
    # horizontal
    if '왼쪽' in lbl:
        horiz = 'Left'
    elif '오른쪽' in lbl:
        horiz = 'Right'
    else:
        horiz = 'Front'
    # vertical
    if surface == 'Inside':
        if ('구개측' in lbl) or ('위' in lbl):
            vert = 'Upper'
        elif ('설측' in lbl) or ('아래' in lbl):
            vert = 'Lower'
        else:
            vert = 'Lower'
    elif surface == 'Occlusal':
        vert = 'N/A'
    else:
        vert = 'N/A'
    return {'surface': surface, 'horizontal': horiz, 'vertical': vert}

def ema_filter(X: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    if X.size == 0: return X
    Y = X.copy()
    for t in range(1, len(X)):
        Y[t] = alpha * Y[t-1] + (1 - alpha) * X[t]
    return Y

def kalman_1d(X: np.ndarray, R: float = 0.01, Q: float = 0.001) -> np.ndarray:
    if X.size == 0: return X
    Z = X.copy()
    nT, nF = Z.shape
    x = Z[0].copy()
    P = np.ones(nF, dtype=np.float32)
    for t in range(1, nT):
        x_pred = x
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x = x_pred + K * (Z[t] - x_pred)
        P = (1 - K) * P_pred
        Z[t] = x
    return Z

def window_stack(X: np.ndarray, L: int, S: int) -> np.ndarray:
    if len(X) < L:
        return np.zeros((0, L, X.shape[-1]), dtype=X.dtype)
    idx = [(s, s + L) for s in range(0, len(X) - L + 1, S)]
    return np.stack([X[s:e] for s, e in idx], axis=0)

# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v1', help='버전 태그 (예: v1, v2)')
    ap.add_argument('--data_root', default=None, help='data_<tag> 루트. 생략 시 자동(data_<tag>)')
    ap.add_argument('--seq_len', type=int, default=30)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--ema_alpha', type=float, default=0.7)
    ap.add_argument('--use_kalman', action='store_true', default=False)
    ap.add_argument('--add_delta', action='store_true', default=False)
    args = ap.parse_args()

    # 입력 루트 자동 결정
    data_root = args.data_root or os.path.join(os.getcwd(), f'data_{args.tag}')
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"입력 폴더가 없습니다: {data_root}")

    # 출력 루트 (버전별 분리)
    out_root = os.path.join('artifacts', args.tag)
    ensure_dir(out_root)

    Xs, ys_surf, ys_h, ys_v = [], [], [], []
    masks_s, masks_h, masks_v = [], [], []
    groups, clips = [], []
    person_set = set()

    for label in KOREAN_LABELS:
        ldir = os.path.join(data_root, label)
        if not os.path.isdir(ldir):
            continue
        for person in sorted(os.listdir(ldir)):
            pdir = os.path.join(ldir, person)
            if not os.path.isdir(pdir):
                continue
            person_set.add(person)
            for npy in sorted(glob.glob(os.path.join(pdir, '*.npy'))):
                X = np.load(npy)
                if len(X) == 0: continue
                if args.ema_alpha and args.ema_alpha > 0:
                    X = ema_filter(X, args.ema_alpha)
                if args.use_kalman:
                    X = kalman_1d(X)
                if args.add_delta:
                    dX = np.zeros_like(X)
                    dX[1:] = X[1:] - X[:-1]
                    X = np.concatenate([X, dX], axis=1)

                W = window_stack(X, args.seq_len, args.stride)
                if len(W) == 0: continue

                lc = label_to_coarse(label)
                y_s = SURFACES.index(lc['surface'])
                y_h = HORIZONTALS.index(lc['horizontal'])
                y_v = -1 if lc['vertical'] == 'N/A' else VERTICALS.index(lc['vertical'])

                Xs.append(W)
                ys_surf.append(np.full((len(W),), y_s, dtype=np.int64))
                ys_h.append(np.full((len(W),), y_h, dtype=np.int64))
                ys_v.append(np.full((len(W),), y_v, dtype=np.int64))
                masks_s.append(np.ones((len(W),), dtype=np.float32))
                masks_h.append(np.ones((len(W),), dtype=np.float32))
                masks_v.append((np.full((len(W),), y_v) != -1).astype(np.float32))

                groups.append(np.full((len(W),), person))
                clip_id = os.path.splitext(os.path.basename(npy))[0]
                clips.append(np.full((len(W),), clip_id))

    if len(Xs) == 0:
        raise RuntimeError('No windows produced. Check data_root or seq_len/stride.')

    X = np.concatenate(Xs, axis=0)
    y_surface = np.concatenate(ys_surf, axis=0)
    y_horizontal = np.concatenate(ys_h, axis=0)
    y_vertical = np.concatenate(ys_v, axis=0)
    mask_surface = np.concatenate(masks_s, axis=0)
    mask_horizontal = np.concatenate(masks_h, axis=0)
    mask_vertical = np.concatenate(masks_v, axis=0)
    groups_arr = np.concatenate(groups, axis=0)
    clips_arr = np.concatenate(clips, axis=0)

    # 저장 경로(버전 폴더)
    np.save(os.path.join(out_root, f'X_data_mt_{args.tag}.npy'), X)
    np.save(os.path.join(out_root, f'y_surface_mt_{args.tag}.npy'), y_surface)
    np.save(os.path.join(out_root, f'y_horizontal_mt_{args.tag}.npy'), y_horizontal)
    np.save(os.path.join(out_root, f'y_vertical_mt_{args.tag}.npy'), y_vertical)
    np.save(os.path.join(out_root, f'mask_surface_mt_{args.tag}.npy'), mask_surface)
    np.save(os.path.join(out_root, f'mask_horizontal_mt_{args.tag}.npy'), mask_horizontal)
    np.save(os.path.join(out_root, f'mask_vertical_mt_{args.tag}.npy'), mask_vertical)
    np.save(os.path.join(out_root, f'groups_mt_{args.tag}.npy'), groups_arr)
    np.save(os.path.join(out_root, f'clips_mt_{args.tag}.npy'), clips_arr)

    # encoders/meta 저장
    meta = {
        'SURFACES': SURFACES,
        'HORIZONTALS': HORIZONTALS,
        'VERTICALS': VERTICALS,
        'persons': sorted(list(person_set)),
        'seq_len': int(args.seq_len),
        'stride': int(args.stride),
        'feat_dim': int(X.shape[-1]),
        'fps': int(args.fps),
        'ema_alpha': float(args.ema_alpha),
        'use_kalman': bool(args.use_kalman),
        'add_delta': bool(args.add_delta),
        'data_root': os.path.abspath(data_root),
        'tag': args.tag
    }
    with open(os.path.join(out_root, f'encoder_meta_{args.tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 매니페스트(재현성)
    with open(os.path.join(out_root, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'tag': args.tag, 'data_root': os.path.abspath(data_root), 'counts': {
            'X': int(X.shape[0]),
            'persons': len(person_set)
        }}, f, ensure_ascii=False, indent=2)

    print(f'✅ Saved to artifacts/{args.tag} | X: {X.shape}')

if __name__ == '__main__':
    main()
