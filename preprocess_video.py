# -*- coding: utf-8 -*-
"""
영상 → 프레임 피처 추출 (MediaPipe FaceMesh + Hands)
- 코끝(nose tip) 원점 정규화
- A) 5th MCP→PIP vs 화면 수평각 (cos,sin)
- B) Wrist→3rd MCP vs 화면 수평각 (cos,sin)
- C) 손 21 랜드마크의 nose 상대좌표 요약 (centroid x,y, spread x,y)
- (옵션) 입/턱 주변 보조 피처
- FPS 리샘플링(고정 15fps), 가려짐/누락 프레임 보정(F-fill)
- 입력 해상도는 640×480 기준으로 처리(다르면 리사이즈)

출력: data_<tag>/<라벨>/<사람>/<클립>.npy (T×F)
"""

import os
import cv2
import glob
import json
import math
import time
import argparse
import numpy as np
from typing import Tuple, Optional

os.environ["PYTHONUTF8"] = "1"

try:
    import mediapipe as mp
except Exception:
    raise RuntimeError("mediapipe가 설치되어 있어야 합니다: pip install mediapipe")

# -------------------- 고정 임계값 --------------------
TARGET_FPS = 15               # 프레임 속도: 15.00 fps
TARGET_W, TARGET_H = 640, 480 # 프레임 너비/높이: 640x480

# -------------------- 라벨 --------------------
KOREAN_LABELS = [
    '왼쪽-협측','중앙-협측','오른쪽-협측',
    '왼쪽-구개측','중앙-구개측','오른쪽-구개측',
    '왼쪽-설측','중앙-설측','오른쪽-설측',
    '오른쪽-위-씹는면','왼쪽-위-씹는면','왼쪽-아래-씹는면','오른쪽-아래-씹는면'
]

# FaceMesh nose tip (468 랜드마크 체계에서 1 인덱스를 nose tip으로 사용)
NOSE_TIP_IDX = 1

# Hand indices (MediaPipe Hands)
WRIST = 0
THIRD_MCP = 9
FIFTH_MCP = 17
FIFTH_PIP = 19

# -------------------- 유틸 --------------------
def angle_to_horizontal(dx: float, dy: float) -> float:
    return math.atan2(dy, dx)

def cos_sin(theta: float) -> Tuple[float, float]:
    return math.cos(theta), math.sin(theta)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_videos(video_root: str):
    for label in KOREAN_LABELS:
        ldir = os.path.join(video_root, label)
        if not os.path.isdir(ldir):
            continue
        for person in sorted(os.listdir(ldir)):
            pdir = os.path.join(ldir, person)
            if not os.path.isdir(pdir):
                continue
            for ext in ("*.mp4","*.mov","*.avi","*.mkv","*.webm"):
                for v in glob.glob(os.path.join(pdir, ext)):
                    if os.path.isdir(v):
                        continue
                    yield label, person, v

def resample_step(in_fps: float, out_fps: float) -> int:
    if in_fps is None or in_fps <= 1:
        return 1
    if out_fps is None or out_fps <= 0:
        return 1
    step = max(1, int(round(in_fps / out_fps)))
    return step

def resolve_video_root(arg_root: Optional[str]) -> str:
    """
    --video_root가 없으면 자동 탐색:
    1) 인자로 받은 경로
    2) 현재 작업폴더의 video_data
    3) 스크립트 파일 기준 video_data
    4) D:\\finalproject\\video_data
    5) 환경변수 VIDEO_ROOT
    """
    candidates = []
    if arg_root: candidates.append(arg_root)
    candidates.append(os.path.join(os.getcwd(), "video_data"))
    try:
        here = os.path.dirname(__file__)
    except NameError:
        here = os.getcwd()
    candidates.append(os.path.join(here, "video_data"))
    candidates.append(r"D:\finalproject\video_data")
    env_root = os.environ.get("VIDEO_ROOT")
    if env_root: candidates.append(env_root)

    tried = []
    for c in candidates:
        if not c:
            continue
        tried.append(c)
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "video_data 폴더를 찾을 수 없습니다.\n"
        + "\n".join(f" - tried: {p}" for p in tried)
        + "\n필요 구조 예) video_data/오른쪽-구개측/P01/클립이름.mp4"
    )

# -------------------- 피처 추출 --------------------
def extract_features_from_video(path: str, use_mouth_feats: bool = True) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    step = resample_step(in_fps, TARGET_FPS)

    face_model = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    hand_model = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.4, min_tracking_confidence=0.4
    )

    T = []
    fi = 0
    last_feat = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 해상도 강제 정규화
        if frame.shape[1] != TARGET_W or frame.shape[0] != TARGET_H:
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

        # FPS 리샘플 (downsample)
        if (fi % step) != 0:
            fi += 1
            continue
        fi += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        f_res = face_model.process(rgb)
        h_res = hand_model.process(rgb)

        # 기본 8차원: 각도 4 + hand xy 요약 4
        feat_dim = 12 if use_mouth_feats else 8
        feat = np.zeros(feat_dim, dtype=np.float32)

        # Face nose tip as origin
        if f_res.multi_face_landmarks:
            face_lm = f_res.multi_face_landmarks[0].landmark
            nx, ny, nz = face_lm[NOSE_TIP_IDX].x, face_lm[NOSE_TIP_IDX].y, face_lm[NOSE_TIP_IDX].z
            if use_mouth_feats:
                mouth_idx = [13, 14]
                mx = float(np.mean([face_lm[i].x for i in mouth_idx]) - nx)
                my = float(np.mean([face_lm[i].y for i in mouth_idx]) - ny)
        else:
            nx, ny, nz = 0.5, 0.5, 0.0
            if use_mouth_feats:
                mx, my = 0.0, 0.0

        # Hand
        if h_res.multi_hand_landmarks:
            hand = h_res.multi_hand_landmarks[0].landmark

            def rel(i):
                return np.array([hand[i].x - nx, hand[i].y - ny, hand[i].z - nz], dtype=np.float32)

            # A) 5th MCP -> PIP 각도
            vA = rel(FIFTH_PIP) - rel(FIFTH_MCP)
            thA = angle_to_horizontal(vA[0], vA[1])
            cA, sA = cos_sin(thA)

            # B) Wrist -> 3rd MCP 각도
            vB = rel(THIRD_MCP) - rel(WRIST)
            thB = angle_to_horizontal(vB[0], vB[1])
            cB, sB = cos_sin(thB)

            feat[0:4] = [cA, sA, cB, sB]

            # C) 21점 상대좌표 요약
            xs = np.array([rel(i)[0] for i in range(21)], dtype=np.float32)
            ys = np.array([rel(i)[1] for i in range(21)], dtype=np.float32)
            cx, cy = float(xs.mean()), float(ys.mean())
            sx, sy = float(xs.std() + 1e-6), float(ys.std() + 1e-6)
            feat[4:8] = [cx, cy, sx, sy]
        else:
            # 손 미검출: 이전 피처 유지(F-fill)
            if last_feat is not None:
                feat[:8] = last_feat[:8]

        if use_mouth_feats:
            feat[8:10] = [mx, my]
            feat[10:12] = [mx, my + 0.05]  # 러프한 턱 추정 보조치

        last_feat = feat.copy()
        T.append(feat)

    cap.release()
    face_model.close()
    hand_model.close()

    if len(T) == 0:
        return np.zeros((0, feat_dim), dtype=np.float32)
    return np.vstack(T)

# -------------------- 메인 --------------------
def main():
    ap = argparse.ArgumentParser()
    # --video_root는 이제 "옵션". 없으면 자동 탐색.
    ap.add_argument('--video_root', default=None, help='video_data 루트 (라벨/사람/*.mp4). 생략 시 자동 탐색')
    ap.add_argument('--tag', default='v1', help='출력 폴더 접미사 (예: v1, v2)')
    ap.add_argument('--add_mouth_feats', action='store_true', default=False, help='입/턱 보조피처 추가')
    args = ap.parse_args()

    # 자동 경로 탐색
    video_root = resolve_video_root(args.video_root)
    print(f"[INFO] video_root = {video_root}")

    out_root = os.path.join(os.getcwd(), f"data_{args.tag}")
    ensure_dir(out_root)

    log = []
    t0 = time.time()
    n_ok, n_fail = 0, 0
    for label, person, vpath in list(list_videos(video_root)):
        try:
            arr = extract_features_from_video(vpath, use_mouth_feats=args.add_mouth_feats)
            save_dir = os.path.join(out_root, label, person)
            ensure_dir(save_dir)
            base = os.path.splitext(os.path.basename(vpath))[0]
            npy_path = os.path.join(save_dir, base + '.npy')
            np.save(npy_path, arr)
            log.append({'label': label, 'person': person, 'video': vpath,
                        'frames': int(arr.shape[0]), 'feat_dim': int(arr.shape[1]),
                        'target_fps': TARGET_FPS, 'size': f'{TARGET_W}x{TARGET_H}'})
            print(f"✅ {npy_path}  {arr.shape}")
            n_ok += 1
        except Exception as e:
            log.append({'label': label, 'person': person, 'video': vpath, 'error': str(e)})
            print(f"⚠️ FAIL {vpath}: {e}")
            n_fail += 1

    meta_dir = os.path.join(out_root, '_meta')
    ensure_dir(meta_dir)
    with open(os.path.join(meta_dir, 'extract_log.json'), 'w', encoding='utf-8') as f:
        json.dump({'items': log, 'sec': time.time() - t0, 'ok': n_ok, 'fail': n_fail}, f, ensure_ascii=False, indent=2)
    print(f'Done. Saved log to {os.path.join(meta_dir, "extract_log.json")} (ok={n_ok}, fail={n_fail})')

if __name__ == '__main__':
    main()
