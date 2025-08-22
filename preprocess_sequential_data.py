# -*- coding: utf-8 -*-
"""
프레임 피처(.npy) → 시퀀스 윈도우 → 멀티태스크 라벨(surface/horizontal/vertical)
- 최종 완성 버전 (프로젝트 폴더 구조 반영):
- 윈도우: seq_len, stride
- EMA / Kalman(옵션) / Δ feature(옵션)
- LOSO를 위한 group(person) / clip ID 저장
- N/A 라벨에 대한 마스크 저장

입력:
  <project_tag>/data_<project_tag>/<라벨>/<사람>/*.npy

출력:
  <project_tag>/artifacts_<project_tag>/<build_tag>/
    X_data_mt_<build_tag>.npy
    ... (기타 라벨, 마스크, 그룹 정보 파일)
"""

# ----- 필수 라이브러리 임포트 -----
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
VERTICALS = ['Lower', 'Upper']

# -------------------- 유틸리티 함수 (Helper Functions) --------------------
def ensure_dir(p: str):
    """경로에 해당하는 폴더가 없으면 생성하여 파일 저장 시 오류를 방지합니다."""
    os.makedirs(p, exist_ok=True)

def label_to_coarse(lbl: str) -> Dict[str, str]:
    """'왼쪽-구개측' 같은 상세한 한글 라벨을 세 가지 대분류 라벨로 변환합니다."""
    if '씹는면' in lbl: surface = 'Occlusal'
    elif ('구개측' in lbl) or ('설측' in lbl): surface = 'Inside'
    else: surface = 'Outside'
    
    if '왼쪽' in lbl: horiz = 'Left'
    elif '오른쪽' in lbl: horiz = 'Right'
    else: horiz = 'Front'
        
    if surface == 'Inside':
        vert = 'Upper' if ('구개측' in lbl) or ('위' in lbl) else 'Lower'
    elif surface == 'Occlusal':
        vert = 'Upper' if '위' in lbl else 'Lower'
    else:
        vert = 'N/A'
    
    return {'surface': surface, 'horizontal': horiz, 'vertical': vert}

def ema_filter(X: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """지수이동평균(EMA) 필터로 시계열 데이터를 부드럽게 만듭니다."""
    if X.size == 0: return X
    Y = X.copy()
    for t in range(1, len(X)):
        Y[t] = alpha * Y[t-1] + (1 - alpha) * X[t]
    return Y

def kalman_1d(X: np.ndarray, R: float = 0.01, Q: float = 0.001) -> np.ndarray:
    """칼만 필터로 노이즈를 제거하고 데이터를 더 정교하게 스무딩합니다."""
    if X.size == 0: return X
    Z = X.copy()
    nT, nF = Z.shape
    x = Z[0].copy()
    P = np.ones(nF, dtype=np.float32)
    for t in range(1, nT):
        x_pred, P_pred = x, P + Q
        K = P_pred / (P_pred + R)
        x = x_pred + K * (Z[t] - x_pred)
        P = (1 - K) * P_pred
        Z[t] = x
    return Z

def window_stack(X: np.ndarray, L: int, S: int) -> np.ndarray:
    """긴 시계열 데이터를 고정된 길이(L)의 작은 조각(window)들로 자릅니다."""
    if len(X) < L:
        return np.zeros((0, L, X.shape[-1]), dtype=X.dtype)
    idx = [(s, s + L) for s in range(0, len(X) - L + 1, S)]
    return np.stack([X[s:e] for s, e in idx], axis=0)

# -------------------- 메인 실행 함수 --------------------
def main():
    ap = argparse.ArgumentParser(description="피처 파일을 모델 학습용 데이터셋으로 가공하는 스크립트")
    # ✅ 수정: --tag를 --project_tag와 --build_tag로 분리
    ap.add_argument('--project_tag', required=True, help='프로젝트 버전 태그 (예: v1_eyes)')
    ap.add_argument('--build_tag', required=True, help='데이터셋 빌드 버전 태그 (예: base_sl15)')
    ap.add_argument('--data_root', default=None, help='입력 피처 데이터 폴더. 생략 시 자동 설정')
    ap.add_argument('--seq_len', type=int, default=30, help="시퀀스 윈도우의 길이")
    ap.add_argument('--stride', type=int, default=5, help="윈도우 이동 간격")
    ap.add_argument('--fps', type=int, default=15, help="데이터의 기준 FPS")
    ap.add_argument('--ema_alpha', type=float, default=0.7, help="EMA 필터 alpha 값 (0이면 비활성화)")
    ap.add_argument('--use_kalman', action='store_true', default=False, help="칼만 필터 사용 여부")
    ap.add_argument('--add_delta', action='store_true', default=False, help="델타 피처(변화량) 추가 여부")
    args = ap.parse_args()

    # ✅ 수정: 입력 경로를 새로운 프로젝트 폴더 구조에 맞게 변경
    data_root = args.data_root or os.path.join(os.getcwd(), args.project_tag, f"data_{args.project_tag}")
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"입력 폴더가 없습니다: {data_root}")
    
    # ✅ 수정: 출력 경로를 새로운 프로젝트 폴더 구조에 맞게 변경
    out_root = os.path.join(os.getcwd(), args.project_tag, f"artifacts_{args.project_tag}", args.build_tag)
    ensure_dir(out_root)

    Xs, ys_surf, ys_h, ys_v = [], [], [], []
    masks_s, masks_h, masks_v = [], [], []
    groups, clips = [], []
    person_set = set()

    print(f"[INFO] Starting build for project '{args.project_tag}', build '{args.build_tag}'")
    for label in KOREAN_LABELS:
        ldir = os.path.join(data_root, label)
        if not os.path.isdir(ldir): continue
        for person in sorted(os.listdir(ldir)):
            pdir = os.path.join(ldir, person)
            if not os.path.isdir(pdir): continue
            person_set.add(person)
            
            for npy in sorted(glob.glob(os.path.join(pdir, '*.npy'))):
                X = np.load(npy)
                if len(X) == 0: continue

                if args.ema_alpha and args.ema_alpha > 0: X = ema_filter(X, args.ema_alpha)
                if args.use_kalman: X = kalman_1d(X)
                if args.add_delta:
                    dX = np.zeros_like(X); dX[1:] = X[1:] - X[:-1]
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
                masks_v.append((np.full((len(W),), y_v) != -1).astype(np.float32))
                groups.append(np.full((len(W),), person))
                clip_id = os.path.splitext(os.path.basename(npy))[0]
                clips.append(np.full((len(W),), clip_id))

    if len(Xs) == 0:
        raise RuntimeError('생성된 윈도우가 없습니다. data_root 또는 seq_len/stride 설정을 확인하세요.')

    X, y_surface, y_horizontal, y_vertical = map(np.concatenate, [Xs, ys_surf, ys_h, ys_v])
    mask_vertical = np.concatenate(masks_v, axis=0)
    groups_arr, clips_arr = map(np.concatenate, [groups, clips])
    
    # ✅ 수정: 저장 파일명에 build_tag 사용
    print(f"[INFO] Saving dataset to {out_root}...")
    np.save(os.path.join(out_root, f'X_data_mt_{args.build_tag}.npy'), X)
    np.save(os.path.join(out_root, f'y_surface_mt_{args.build_tag}.npy'), y_surface)
    np.save(os.path.join(out_root, f'y_horizontal_mt_{args.build_tag}.npy'), y_horizontal)
    np.save(os.path.join(out_root, f'y_vertical_mt_{args.build_tag}.npy'), y_vertical)
    np.save(os.path.join(out_root, f'mask_vertical_mt_{args.build_tag}.npy'), mask_vertical)
    np.save(os.path.join(out_root, f'groups_mt_{args.build_tag}.npy'), groups_arr)
    np.save(os.path.join(out_root, f'clips_mt_{args.build_tag}.npy'), clips_arr)

    # ✅ 수정: 메타데이터에 두 태그 모두 기록
    meta = {
        'SURFACES': SURFACES, 'HORIZONTALS': HORIZONTALS, 'VERTICALS': VERTICALS,
        'persons': sorted(list(person_set)),
        'seq_len': int(args.seq_len), 'stride': int(args.stride),
        'feat_dim': int(X.shape[-1]), 'fps': int(args.fps),
        'ema_alpha': float(args.ema_alpha), 'use_kalman': bool(args.use_kalman),
        'add_delta': bool(args.add_delta),
        'project_tag': args.project_tag, 'build_tag': args.build_tag,
        'data_root': os.path.abspath(data_root)
    }
    with open(os.path.join(out_root, f'encoder_meta_{args.build_tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved to {out_root} | X shape: {X.shape}")

if __name__ == '__main__':
    main()