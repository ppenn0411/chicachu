# -*- coding: utf-8 -*-
"""
영상 → 프레임 피처 추출 (MediaPipe FaceMesh + Hands)
- 최종 완성 버전 (프로젝트 폴더 구조 반영):
- 코끝(nose tip) 원점 정규화
- A) 5th MCP→PIP vs 화면 수평각 (cos,sin)
- B) Wrist→3rd MCP vs 화면 수평각 (cos,sin)
- C) 손 21 랜드마크의 nose 상대좌표 요약 (centroid x,y, spread x,y)
- (옵션) 코+양쪽 눈 '기준 삼각형' 보조 피처
- FPS 리샘플링(고정 15fps), 시간적 평균화(Temporal Averaging)로 정보손실 최소화
- 입력 해상도는 640×480 기준으로 처리(레터박스 적용으로 비율 유지)
- 얼굴 미검출 프레임은 노이즈로 간주하여 무시 (데이터 품질 보증)
- 손 미검출 시 Zero-fill 적용 (동작 없음을 명확히 신호)

출력: <project_tag>/data_<project_tag>/<라벨>/<사람>/<클립>.npy (T×F)
"""

# ----- 필수 라이브러리 임포트 -----
import os
import cv2
import glob
import json
import math
import time
import argparse
import numpy as np
from typing import Tuple, Optional

# 시스템 인코딩 설정 (Windows 환경에서의 한글 경로 문제 방지)
os.environ["PYTHONUTF8"] = "1"

# MediaPipe 라이브러리 임포트 및 예외 처리
try:
    import mediapipe as mp
except Exception:
    raise RuntimeError("mediapipe가 설치되어 있어야 합니다: pip install mediapipe")

# -------------------- 고정 임계값 (Constants) --------------------
TARGET_FPS = 15
TARGET_W, TARGET_H = 640, 480

# -------------------- 데이터셋 관련 설정 --------------------
KOREAN_LABELS = [
    '왼쪽-협측','중앙-협측','오른쪽-협측',
    '왼쪽-구개측','중앙-구개측','오른쪽-구개측',
    '왼쪽-설측','중앙-설측','오른쪽-설측',
    '오른쪽-위-씹는면','왼쪽-위-씹는면','왼쪽-아래-씹는면','오른쪽-아래-씹는면'
]

# MediaPipe 랜드마크 인덱스
NOSE_TIP_IDX = 1
LEFT_EYE_IDX = 133
RIGHT_EYE_IDX = 362

# Hand indices (MediaPipe Hands)
WRIST = 0
THIRD_MCP = 9
FIFTH_MCP = 17
FIFTH_PIP = 19

# -------------------- 유틸리티 함수 (Helper Functions) --------------------
def angle_to_horizontal(dx: float, dy: float) -> float:
    """x, y 변화량으로 수평축과의 각도(라디안)를 계산합니다."""
    return math.atan2(dy, dx)

def cos_sin(theta: float) -> Tuple[float, float]:
    """각도를 (cos, sin) 쌍으로 변환합니다. 모델이 0도와 360도를 같은 지점으로 인식하도록 돕습니다."""
    return math.cos(theta), math.sin(theta)

def ensure_dir(p: str):
    """경로에 해당하는 폴더가 없으면 생성하여 파일 저장 시 오류를 방지합니다."""
    os.makedirs(p, exist_ok=True)

def list_videos(video_root: str):
    """지정된 경로 구조에 따라 모든 비디오 파일의 경로를 재귀적으로 찾습니다."""
    for label in KOREAN_LABELS:
        ldir = os.path.join(video_root, label)
        if not os.path.isdir(ldir): continue
        for person in sorted(os.listdir(ldir)):
            pdir = os.path.join(ldir, person)
            if not os.path.isdir(pdir): continue
            for ext in ("*.mp4","*.mov","*.avi","*.mkv","*.webm"):
                for v in glob.glob(os.path.join(pdir, ext)):
                    if os.path.isdir(v): continue
                    yield label, person, v

def resample_step(in_fps: float, out_fps: float) -> int:
    """
    시간적 평균화를 위해 몇 개의 프레임을 하나의 그룹으로 묶을지(step) 계산합니다.
    예: 60fps -> 15fps 변환 시, 60/15=4. 즉, 4개 프레임을 평균내어 1개의 대표 프레임을 만듭니다.
    """
    if in_fps is None or in_fps <= 1: return 1
    if out_fps is None or out_fps <= 0: return 1
    return max(1, int(round(in_fps / out_fps)))

def resolve_video_root(arg_root: Optional[str]) -> str:
    """사용자가 --video_root를 지정하지 않았을 경우, 여러 후보 경로에서 비디오 폴더를 자동으로 탐색하는 편의 기능입니다."""
    candidates = []
    if arg_root: candidates.append(arg_root)
    candidates.append(os.path.join(os.getcwd(), "video_data"))
    try:
        here = os.path.dirname(__file__)
    except NameError:
        here = os.getcwd()
    candidates.append(os.path.join(here, "video_data"))
    candidates.append(r"D:\chicachu\video_data") # 이 부분은 개인 환경에 맞게 수정
    env_root = os.environ.get("VIDEO_ROOT")
    if env_root: candidates.append(env_root)
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError("video_data 폴더를 찾을 수 없습니다.")

# -------------------- 핵심 기능: 피처 추출 함수 --------------------
def extract_features_from_video(path: str, use_eye_feats: bool = True) -> np.ndarray:
    """
    단일 비디오 파일로부터 시계열 특징(feature)을 추출합니다.
    레터박싱, 시간적 평균화, 노이즈 제거 등 데이터 정제 과정을 포함합니다.
    """
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
    feature_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        if h == TARGET_H and w == TARGET_W:
            padded_frame = frame
        else:
            scale = min(TARGET_W / w, TARGET_H / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded_frame = np.full((TARGET_H, TARGET_W, 3), 0, dtype=np.uint8)
            top = (TARGET_H - new_h) // 2
            left = (TARGET_W - new_w) // 2
            padded_frame[top:top + new_h, left:left + new_w] = resized

        rgb = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        f_res = face_model.process(rgb)

        if not f_res.multi_face_landmarks:
            continue

        h_res = hand_model.process(rgb)
        feat_dim = 12 if use_eye_feats else 8
        feat = np.zeros(feat_dim, dtype=np.float32)

        face_lm = f_res.multi_face_landmarks[0].landmark
        nx, ny, _ = face_lm[NOSE_TIP_IDX].x, face_lm[NOSE_TIP_IDX].y, face_lm[NOSE_TIP_IDX].z
        if use_eye_feats:
            lex = face_lm[LEFT_EYE_IDX].x - nx
            ley = face_lm[LEFT_EYE_IDX].y - ny
            rex = face_lm[RIGHT_EYE_IDX].x - nx
            rey = face_lm[RIGHT_EYE_IDX].y - ny

        if h_res.multi_hand_landmarks:
            hand = h_res.multi_hand_landmarks[0].landmark
            def rel(i):
                return np.array([hand[i].x - nx, hand[i].y - ny], dtype=np.float32)

            vA = rel(FIFTH_PIP) - rel(FIFTH_MCP)
            thA = angle_to_horizontal(vA[0], vA[1])
            cA, sA = cos_sin(thA)
            vB = rel(THIRD_MCP) - rel(WRIST)
            thB = angle_to_horizontal(vB[0], vB[1])
            cB, sB = cos_sin(thB)
            feat[0:4] = [cA, sA, cB, sB]

            xs = np.array([rel(i)[0] for i in range(21)], dtype=np.float32)
            ys = np.array([rel(i)[1] for i in range(21)], dtype=np.float32)
            cx, cy = float(xs.mean()), float(ys.mean())
            sx, sy = float(xs.std() + 1e-6), float(ys.std() + 1e-6)
            feat[4:8] = [cx, cy, sx, sy]
        else:
            pass

        if use_eye_feats:
            feat[8:12] = [lex, ley, rex, rey]
        
        feature_buffer.append(feat)
        if len(feature_buffer) == step:
            avg_feat = np.mean(np.array(feature_buffer), axis=0)
            T.append(avg_feat)
            feature_buffer = []

    if feature_buffer:
        avg_feat = np.mean(np.array(feature_buffer), axis=0)
        T.append(avg_feat)

    cap.release()
    face_model.close()
    hand_model.close()

    if len(T) == 0:
        return np.zeros((0, feat_dim), dtype=np.float32)
    return np.vstack(T)

# -------------------- 메인 실행 함수 --------------------
def main():
    ap = argparse.ArgumentParser(description="영상으로부터 손/얼굴 특징을 추출하는 스크립트")
    ap.add_argument('--video_root', default=None, help='원본 비디오가 담긴 video_data 루트')
    ap.add_argument('--project_tag', required=True, help='프로젝트 버전 태그 (예: v1_eyes)')
    ap.add_argument('--add_eye_feats', action='store_true', default=False, help='코+양쪽 눈 보조피처 추가')
    args = ap.parse_args()

    video_root = resolve_video_root(args.video_root)
    print(f"[INFO] Source video root = {video_root}")

    out_root = os.path.join(os.getcwd(), args.project_tag, f"data_{args.project_tag}")
    ensure_dir(out_root)
    print(f"[INFO] Output feature root = {out_root}")

    log = []
    t0 = time.time()
    n_ok, n_fail = 0, 0
    video_list = list(list_videos(video_root))
    print(f"[INFO] Found {len(video_list)} videos to process.")

    for i, (label, person, vpath) in enumerate(video_list):
        print(f"[INFO] Processing video {i+1}/{len(video_list)}: {vpath}")
        try:
            arr = extract_features_from_video(vpath, use_eye_feats=args.add_eye_feats)
            
            if arr.shape[0] == 0:
                print(f"⚠️ SKIP {vpath}: No valid frames found.")
                log.append({'label': label, 'person': person, 'video': vpath, 'error': 'No valid frames'})
                n_fail += 1
                continue

            save_dir = os.path.join(out_root, label, person)
            ensure_dir(save_dir)
            base = os.path.splitext(os.path.basename(vpath))[0]
            npy_path = os.path.join(save_dir, base + '.npy')
            np.save(npy_path, arr)
            
            log.append({'label': label, 'person': person, 'video': vpath,
                        'frames': int(arr.shape[0]),
                        'feat_dim': int(arr.shape[1]),
                        'target_fps': TARGET_FPS, 'size': f'{TARGET_W}x{TARGET_H}'})
            n_ok += 1
        except Exception as e:
            log.append({'label': label, 'person': person, 'video': vpath, 'error': str(e)})
            print(f"⚠️ FAIL {vpath}: {e}")
            n_fail += 1

    # ✅ 수정: 모든 작업 완료 후, 버전별 로그를 별도의 logs 폴더에 저장
    # 1. 로그 저장 전용 폴더 생성
    log_dir = os.path.join(os.getcwd(), 'logs')
    ensure_dir(log_dir)
    
    # 2. 프로젝트 태그를 포함한 로그 파일 경로 설정
    log_filepath = os.path.join(log_dir, f'{args.project_tag}_extraction_log.json')
    
    # 3. 로그 파일 저장
    with open(log_filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'project_tag': args.project_tag,
            'items': log, 
            'sec': time.time() - t0, 
            'ok': n_ok, 
            'fail': n_fail
        }, f, ensure_ascii=False, indent=2)
        
    print(f'\nDone. Saved log to {log_filepath} (ok={n_ok}, fail={n_fail})')


if __name__ == '__main__':
    main()