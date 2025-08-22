# -*- coding: utf-8 -*-
"""
Random Forest 모델 학습 + LOSO 평가 (프로젝트 폴더 구조 및 시간 로깅 기능 반영)
- 입력: artifacts/<project_tag>/<build_tag> 의 데이터
- 시계열 데이터를 통계 피처로 변환하여 2D 데이터로 가공
- 3개의 Task에 대해 3개의 독립적인 RF 모델을 학습
- 출력: runs/<project_tag>/<build_tag>/rf_<timestamp>/... 에 결과 저장
"""

import os
import json
import argparse
import time
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

# ----- 데이터 로딩 및 LOSO 분할 함수 (TCN 스크립트와 동일) -----
def load_arrays(root_path, build_tag):
    """artifacts/<project_tag>/<build_tag> 에서 데이터 로드"""
    X = np.load(os.path.join(root_path, f'X_data_mt_{build_tag}.npy'))
    y_s = np.load(os.path.join(root_path, f'y_surface_mt_{build_tag}.npy'))
    y_h = np.load(os.path.join(root_path, f'y_horizontal_mt_{build_tag}.npy'))
    y_v = np.load(os.path.join(root_path, f'y_vertical_mt_{build_tag}.npy'))
    m_v = np.load(os.path.join(root_path, f'mask_vertical_mt_{build_tag}.npy'))
    groups = np.load(os.path.join(root_path, f'groups_mt_{build_tag}.npy'))
    meta = json.load(open(os.path.join(root_path, f'encoder_meta_{build_tag}.json'),'r',encoding='utf-8'))
    return X, y_s, y_h, y_v, m_v, groups, meta

def make_loso_splits(groups):
    """사람(group)별로 LOSO 교차 검증을 위한 데이터 인덱스를 분할합니다."""
    persons = sorted(list(set(groups.tolist())))
    for p in persons:
        test_idx = (groups == p)
        train_idx = ~test_idx
        yield p, train_idx, test_idx

# ----- 핵심: 시계열 데이터를 2D 통계 피처로 변환하는 함수 -----
def featurize_sequences(X: np.ndarray) -> np.ndarray:
    """ (N, L, F) -> (N, F*4) 형태로 변환. 각 피처의 평균, 표준편차, 최소, 최대값을 계산. """
    if X.ndim != 3:
        raise ValueError("입력 배열은 3차원이어야 합니다 (N, L, F).")
    N, L, F = X.shape
    stat_features = np.zeros((N, F * 4), dtype=np.float32)
    for f_idx in range(F):
        stat_features[:, f_idx * 4 + 0] = np.mean(X[:, :, f_idx], axis=1)
        stat_features[:, f_idx * 4 + 1] = np.std(X[:, :, f_idx], axis=1)
        stat_features[:, f_idx * 4 + 2] = np.min(X[:, :, f_idx], axis=1)
        stat_features[:, f_idx * 4 + 3] = np.max(X[:, :, f_idx], axis=1)
    return stat_features

# -------------------- 메인 실행 함수 --------------------
def main():
    ap = argparse.ArgumentParser(description="Random Forest 모델 학습 및 LOSO 평가 스크립트")
    ap.add_argument('--project_tag', required=True, help='프로젝트 버전 태그 (예: v1_eyes)')
    ap.add_argument('--build_tag', required=True, help='데이터셋 빌드 버전 태그 (예: base_sl15)')
    ap.add_argument('--n_estimators', type=int, default=200, help="랜덤 포레스트의 트리 개수")
    ap.add_argument('--max_depth', type=int, default=15, help="트리의 최대 깊이")
    args = ap.parse_args()

    # 경로 설정
    artifacts_root = os.path.join('artifacts', args.project_tag, args.build_tag)
    if not os.path.isdir(artifacts_root):
        raise FileNotFoundError(f"입력 배열 폴더가 없습니다: {artifacts_root}")

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    runs_root = os.path.join('runs', args.project_tag, args.build_tag, f'rf_{timestamp}')
    os.makedirs(runs_root, exist_ok=True)
    
    # 데이터 로딩 및 2D 피처로 가공
    X_seq, y_s, y_h, y_v, m_v, groups, meta = load_arrays(artifacts_root, args.build_tag)
    print(f"[INFO] Original sequence shape: {X_seq.shape}")
    X_flat = featurize_sequences(X_seq)
    print(f"[INFO] Featurized flat shape: {X_flat.shape}")
    
    # 전체 실행 정보 저장 (manifest.json)
    with open(os.path.join(runs_root, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'project_tag': args.project_tag, 'build_tag': args.build_tag, 'model_type': 'RandomForest',
            'timestamp': timestamp, 'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
            'artifacts_root': os.path.abspath(artifacts_root)
        }, f, ensure_ascii=False, indent=2)

    # LOSO 교차 검증 루프
    for fold_i, (person, tr_idx, te_idx) in enumerate(make_loso_splits(groups), start=1):
        t_fold_start = time.time()

        fold_dir = os.path.join(runs_root, f'loso-{person}')
        for d in ['models', 'preds', 'metrics']:
            os.makedirs(os.path.join(fold_dir, d), exist_ok=True)
        print(f'\n=== FOLD {fold_i}/{len(set(groups))} | Testing on person={person} ===')

        Xtr, Xte = X_flat[tr_idx], X_flat[te_idx]
        
        # --- Task 1: Surface ---
        ytr_s, yte_s = y_s[tr_idx], y_s[te_idx]
        print("[INFO] Training 'surface' model...")
        model_s = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42, n_jobs=-1)
        model_s.fit(Xtr, ytr_s)
        ypred_s = model_s.predict(Xte)
        
        # --- Task 2: Horizontal ---
        ytr_h, yte_h = y_h[tr_idx], y_h[te_idx]
        print("[INFO] Training 'horizontal' model...")
        model_h = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42, n_jobs=-1)
        model_h.fit(Xtr, ytr_h)
        ypred_h = model_h.predict(Xte)
        
        # --- Task 3: Vertical (마스크가 1인 데이터만 사용) ---
        mtr_v, mte_v = m_v[tr_idx], m_v[te_idx]
        ytr_v, yte_v = y_v[tr_idx], y_v[te_idx]
        print("[INFO] Training 'vertical' model...")
        model_v = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42, n_jobs=-1)
        if np.any(mtr_v == 1):
            model_v.fit(Xtr[mtr_v == 1], ytr_v[mtr_v == 1])
        ypred_v = np.full_like(yte_v, -1)
        if np.any(mte_v == 1) and hasattr(model_v, 'classes_'):
            ypred_v[mte_v == 1] = model_v.predict(Xte[mte_v == 1])

        # --- [생략되지 않은 부분] 결과 저장 ---
        # 모델 저장
        pickle.dump(model_s, open(os.path.join(fold_dir, 'models', 'rf_surface.pkl'), 'wb'))
        pickle.dump(model_h, open(os.path.join(fold_dir, 'models', 'rf_horizontal.pkl'), 'wb'))
        if hasattr(model_v, 'classes_'):
             pickle.dump(model_v, open(os.path.join(fold_dir, 'models', 'rf_vertical.pkl'), 'wb'))
        
        # 예측 및 정답 저장
        preds_dir = os.path.join(fold_dir, 'preds')
        np.save(os.path.join(preds_dir, 'y_true_surface.npy'), yte_s)
        np.save(os.path.join(preds_dir, 'y_pred_surface.npy'), ypred_s)
        np.save(os.path.join(preds_dir, 'y_true_horizontal.npy'), yte_h)
        np.save(os.path.join(preds_dir, 'y_pred_horizontal.npy'), ypred_h)
        np.save(os.path.join(preds_dir, 'y_true_vertical.npy'), yte_v)
        np.save(os.path.join(preds_dir, 'y_pred_vertical.npy'), ypred_v)

        # 성능 지표 저장
        metrics_dir = os.path.join(fold_dir, 'metrics')
        rep_s = classification_report(yte_s, ypred_s, output_dict=True, zero_division=0)
        rep_h = classification_report(yte_h, ypred_h, output_dict=True, zero_division=0)
        f1_metrics = {
            'surface': {'macro_f1': rep_s['macro avg']['f1-score']},
            'horizontal': {'macro_f1': rep_h['macro avg']['f1-score']}
        }
        if np.any(mte_v == 1):
             rep_v = classification_report(yte_v[mte_v == 1], ypred_v[mte_v == 1], output_dict=True, zero_division=0)
             f1_metrics['vertical'] = {'macro_f1': rep_v['macro avg']['f1-score']}
        
        with open(os.path.join(metrics_dir, 'f1.json'), 'w', encoding='utf-8') as f:
            json.dump(f1_metrics, f, ensure_ascii=False, indent=2)

        # Fold 처리 시간 기록 및 저장
        fold_duration_sec = time.time() - t_fold_start
        with open(os.path.join(fold_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'person': person,
                'train_count': int(tr_idx.sum()),
                'test_count': int(te_idx.sum()),
                'duration_sec': fold_duration_sec
            }, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Fold {person} complete in {fold_duration_sec:.2f}s. Val F1(surface): {f1_metrics['surface']['macro_f1']:.4f}")

    print('\n✅ All folds finished for Random Forest.')
    print(f'📁 Runs saved to: {runs_root}')

if __name__ == '__main__':
    main()