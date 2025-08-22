# -*- coding: utf-8 -*-
"""
LSTM 모델 학습 + LOSO 평가 (프로젝트 폴더 구조 및 시간 로깅 기능 반영)
- 입력: artifacts/<project_tag>/<build_tag> 의 데이터
- TCN과 동일한 멀티태스크, LOSO 평가 로직 사용
- Zero-fill된 데이터를 무시하기 위한 Masking 레이어 추가
- 출력: runs/<project_tag>/<build_tag>/lstm_<timestamp>/... 에 결과 저장
"""

import os
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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

# -------------------- 모델 구축 함수 --------------------
def build_mt_lstm(input_shape: tuple, n_surface=3, n_h=3, n_v=2):
    """멀티태스크 LSTM 모델을 구축합니다."""
    inp = layers.Input(shape=input_shape)
    
    # Masking 레이어: 입력값 중 모든 피처가 0.0인 타임스텝은 계산에서 제외시킵니다.
    # Zero-fill된 손 미검출 프레임을 모델이 무시하도록 하는 효과가 있습니다.
    x = layers.Masking(mask_value=0.0)(inp)
    
    # LSTM 레이어를 2겹으로 쌓아 시간적 패턴을 학습합니다.
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(128)(x)
    
    # 멀티태스크 출력을 위한 Dense 레이어들
    h = layers.Dense(128, activation='relu')(x)
    out_s = layers.Dense(n_surface, activation='softmax', name='surface')(h)
    out_h = layers.Dense(n_h, activation='softmax', name='horizontal')(h)
    out_v = layers.Dense(n_v, activation='softmax', name='vertical')(h)
    
    model = models.Model(inp, [out_s, out_h, out_v])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "surface": "sparse_categorical_crossentropy",
            "horizontal": "sparse_categorical_crossentropy",
            "vertical": "sparse_categorical_crossentropy"
        },
        metrics=["accuracy"]
    )
    return model

# -------------------- 메인 실행 함수 --------------------
def main():
    ap = argparse.ArgumentParser(description="LSTM 모델 학습 및 LOSO 평가 스크립트")
    ap.add_argument('--project_tag', required=True, help='프로젝트 버전 태그 (예: v1_eyes)')
    ap.add_argument('--build_tag', required=True, help='데이터셋 빌드 버전 태그 (예: base_sl15)')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    # 경로 설정
    artifacts_root = os.path.join('artifacts', args.project_tag, args.build_tag)
    if not os.path.isdir(artifacts_root):
        raise FileNotFoundError(f"입력 배열 폴더가 없습니다: {artifacts_root}")

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    runs_root = os.path.join('runs', args.project_tag, args.build_tag, f'lstm_{timestamp}')
    os.makedirs(runs_root, exist_ok=True)

    # 데이터 로딩
    X, y_s, y_h, y_v, m_v, groups, meta = load_arrays(artifacts_root, args.build_tag)
    N, L, F = X.shape
    
    # 전체 실행 정보 저장 (manifest.json)
    with open(os.path.join(runs_root, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'project_tag': args.project_tag, 'build_tag': args.build_tag, 'model_type': 'LSTM',
            'timestamp': timestamp, 'seq_len': int(L), 'feat_dim': int(F),
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'artifacts_root': os.path.abspath(artifacts_root)
        }, f, ensure_ascii=False, indent=2)

    # LOSO 교차 검증 루프
    for fold_i, (person, tr_idx, te_idx) in enumerate(make_loso_splits(groups), start=1):
        t_fold_start = time.time()

        fold_dir = os.path.join(runs_root, f'loso-{person}')
        for d in ['models', 'preds', 'metrics']:
            os.makedirs(os.path.join(fold_dir, d), exist_ok=True)

        print(f'\n=== FOLD {fold_i}/{len(set(groups))} | Testing on person={person} ===')

        # 데이터 분할
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr_s, yte_s = y_s[tr_idx], y_s[te_idx]
        ytr_h, yte_h = y_h[tr_idx], y_h[te_idx]
        ytr_v, yte_v = y_v[tr_idx], y_v[te_idx]
        mtr_v, mte_v = m_v[tr_idx], m_v[te_idx]
        ytr_v_eff = np.where(ytr_v < 0, 0, ytr_v)
        yte_v_eff = np.where(yte_v < 0, 0, yte_v)
        
        # 모델 생성 및 학습
        model = build_mt_lstm((L, F))
        cb = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(fold_dir, 'models', 'lstm_best_model.h5'),
                save_best_only=True, monitor='val_surface_accuracy', mode='max'),
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]

        model.fit(
            Xtr, [ytr_s, ytr_h, ytr_v_eff],
            validation_data=(Xte, [yte_s, yte_h, yte_v_eff], {"vertical": mte_v}),
            epochs=args.epochs, batch_size=args.batch_size,
            sample_weight={"surface": np.ones_like(ytr_s), "horizontal": np.ones_like(ytr_h), "vertical": mtr_v},
            callbacks=cb, verbose=2
        )

        # 예측 및 결과 저장
        yhat_s, yhat_h, yhat_v = model.predict(Xte, verbose=0)
        ypred_s, ypred_h, ypred_v = yhat_s.argmax(1), yhat_h.argmax(1), yhat_v.argmax(1)
        np.save(os.path.join(fold_dir, 'preds', 'y_true_surface.npy'), yte_s)
        np.save(os.path.join(fold_dir, 'preds', 'y_pred_surface.npy'), ypred_s)
        np.save(os.path.join(fold_dir, 'preds', 'y_true_horizontal.npy'), yte_h)
        np.save(os.path.join(fold_dir, 'preds', 'y_pred_horizontal.npy'), ypred_h)
        np.save(os.path.join(fold_dir, 'preds', 'y_true_vertical.npy'), yte_v)
        np.save(os.path.join(fold_dir, 'preds', 'y_pred_vertical.npy'), ypred_v)

        # 성능 지표 계산 및 저장
        metrics_dir = os.path.join(fold_dir, 'metrics')
        rep_s = classification_report(yte_s, ypred_s, output_dict=True, zero_division=0)
        rep_h = classification_report(yte_h, ypred_h, output_dict=True, zero_division=0)
        f1_metrics = {
            'surface': {'macro_f1': rep_s['macro avg']['f1-score']},
            'horizontal': {'macro_f1': rep_h['macro avg']['f1-score']}
        }
        if np.any(yte_v >= 0):
            mask = yte_v >= 0
            rep_v = classification_report(yte_v[mask], ypred_v[mask], output_dict=True, zero_division=0)
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

    print('\n✅ All folds finished for LSTM.')
    print(f'📁 Runs saved to: {runs_root}')

if __name__ == '__main__':
    main()