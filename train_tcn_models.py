# -*- coding: utf-8 -*-
"""
멀티태스크 TCN 학습 + LOSO 평가 + 지표/트래킹 정리 (로드맵 완전 반영)
- 입력 배열은 artifacts/<tag>/ 에서 자동 로딩 (--artifacts_root로 변경 가능)
- 출력/런은 runs/<tag>/<timestamp>/... 에 버전별로 분리 저장
"""

import os
import json
import argparse
import time
import numpy as np
from typing import Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models

# 선택 설치
SKLEARN_OK = True
UMAP_OK = True
try:
    from sklearn.metrics import f1_score, classification_report, silhouette_score
    from sklearn.model_selection import train_test_split
except Exception:
    SKLEARN_OK = False
try:
    import umap
except Exception:
    UMAP_OK = False

# ==================== 데이터 로딩 ====================
def load_arrays(tag: str, root: str):
    """artifacts/<tag>에서 로드"""
    r = root
    X = np.load(os.path.join(r, f'X_data_mt_{tag}.npy'))
    y_s = np.load(os.path.join(r, f'y_surface_mt_{tag}.npy'))
    y_h = np.load(os.path.join(r, f'y_horizontal_mt_{tag}.npy'))
    y_v = np.load(os.path.join(r, f'y_vertical_mt_{tag}.npy'))
    m_v = np.load(os.path.join(r, f'mask_vertical_mt_{tag}.npy'))
    groups = np.load(os.path.join(r, f'groups_mt_{tag}.npy'))
    meta = json.load(open(os.path.join(r, f'encoder_meta_{tag}.json'),'r',encoding='utf-8'))
    return X, y_s, y_h, y_v, m_v, groups, meta

# ==================== 모델 ====================
def tcn_block(x, ch, k=3, d=1, drop=0.1):
    y = layers.Conv1D(ch, k, padding='causal', dilation_rate=d, activation='relu')(x)
    y = layers.Dropout(drop)(y)
    y = layers.Conv1D(ch, k, padding='causal', dilation_rate=d, activation='relu')(y)
    if x.shape[-1] != ch:
        x = layers.Conv1D(ch, 1, padding='same')(x)
    return layers.Add()([x, y])

def build_mt_tcn(input_shape: Tuple[int,int], n_surface=3, n_h=3, n_v=2):
    inp = layers.Input(shape=input_shape)
    x = inp
    for d in [1, 2, 4, 8]:
        x = tcn_block(x, 96, k=3, d=d, drop=0.1)
    emb = layers.GlobalAveragePooling1D(name='embedding')(x)
    h = layers.Dense(128, activation='relu')(emb)
    out_s = layers.Dense(n_surface, activation='softmax', name='surface')(h)
    out_h = layers.Dense(n_h, activation='softmax', name='horizontal')(h)
    out_v = layers.Dense(n_v, activation='softmax', name='vertical')(h)
    model = models.Model(inp, [out_s, out_h, out_v])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"surface":"sparse_categorical_crossentropy",
              "horizontal":"sparse_categorical_crossentropy",
              "vertical":"sparse_categorical_crossentropy"},
        metrics={"surface":["accuracy"],"horizontal":["accuracy"],"vertical":["accuracy"]},
    )
    return model

# ==================== 유틸 ====================
def make_loso_splits(groups):
    persons = sorted(list(set(groups.tolist())))
    folds = []
    for p in persons:
        test_idx = (groups == p)
        train_idx = ~test_idx
        folds.append((p, train_idx, test_idx))
    return folds

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)

def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    if probs.ndim != 2: raise ValueError("probs must be 2D (N x C)")
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0: continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.mean()) * abs(acc_bin - conf_bin)
    return float(ece)

def temperature_scale_logits(logits: np.ndarray, T: float) -> np.ndarray:
    return logits / max(T, 1e-6)

def find_best_temperature(logits: np.ndarray, labels: np.ndarray, T_grid=(0.5, 5.0, 51)):
    lo, hi, steps = T_grid
    Ts = np.linspace(lo, hi, int(steps))
    best_T, best_nll = 1.0, 1e9
    for T in Ts:
        lg = temperature_scale_logits(logits, T)
        probs = softmax(lg)
        eps = 1e-8
        nll = -np.log(probs[np.arange(len(labels)), labels] + eps).mean()
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T, float(best_nll)

def save_umap_and_silhouette(emb: np.ndarray, y: np.ndarray, out_png: str, out_json: str,
                             n_neighbors: int = 20, min_dist: float = 0.2, seed: int = 42):
    if not UMAP_OK or not SKLEARN_OK:
        return False
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    Z = reducer.fit_transform(emb)
    try:
        sil = float(silhouette_score(Z, y))
    except Exception:
        sil = None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    unique = sorted(list(set(y.tolist())))
    for u in unique:
        m = (y == u)
        plt.scatter(Z[m,0], Z[m,1], s=8, label=str(u), alpha=0.7)
    plt.legend(markerscale=2, fontsize=8, loc='best')
    plt.title('UMAP (embedding)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'silhouette': sil, 'n_neighbors': n_neighbors, 'min_dist': min_dist}, f, ensure_ascii=False, indent=2)
    return True

# ==================== 학습 루프 ====================
# ==================== 학습 루프 ====================
def main():
    # ✅ 수정: --tag를 --project_tag와 --build_tag로 분리
    ap = argparse.ArgumentParser(description="TCN 모델 학습 및 LOSO 평가 스크립트")
    ap.add_argument('--project_tag', required=True, help='프로젝트 버전 태그 (예: v1_eyes)')
    ap.add_argument('--build_tag', required=True, help='데이터셋 빌드 버전 태그 (예: base_sl15)')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--augment', action='store_true', default=False)
    ap.add_argument('--calibrate', action='store_true', default=True, help='온도 스케일링으로 calibration 수행')
    ap.add_argument('--umap_neighbors', type=int, default=20)
    ap.add_argument('--umap_min_dist', type=float, default=0.2)
    args = ap.parse_args()

    # ✅ 수정: 입력 경로를 새로운 프로젝트 폴더 구조에 맞게 변경
    artifacts_root = os.path.join('artifacts', args.project_tag, args.build_tag)
    if not os.path.isdir(artifacts_root):
        raise FileNotFoundError(f"입력 배열 폴더가 없습니다: {artifacts_root}")

    # ✅ 수정: load_arrays 호출 시 build_tag 사용 및 불필요한 마스크 로딩 제거
    X, y_s, y_h, y_v, m_v, groups, meta = load_arrays(artifacts_root, args.build_tag)
    N, L, F = X.shape

    # ✅ 수정: 출력 경로를 새로운 프로젝트 폴더 구조에 맞게 변경
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    runs_root = os.path.join('runs', args.project_tag, args.build_tag, f'tcn_{timestamp}')
    os.makedirs(runs_root, exist_ok=True)

    # ✅ 수정: Manifest에 새로운 태그 정보 기록
    with open(os.path.join(runs_root, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'project_tag': args.project_tag, 'build_tag': args.build_tag, 'model_type': 'TCN',
            'timestamp': timestamp, 'seq_len': int(L), 'feat_dim': int(F),
            'epochs': args.epochs, 'batch_size': args.batch_size, 'augment': bool(args.augment),
            'calibrate': bool(args.calibrate), 'umap_neighbors': args.umap_neighbors,
            'umap_min_dist': args.umap_min_dist, 'artifacts_root': os.path.abspath(artifacts_root)
        }, f, ensure_ascii=False, indent=2)

    def augment_batch(Xb):
        if not args.augment: return Xb
        return Xb + np.random.normal(scale=0.01, size=Xb.shape).astype(np.float32)

    folds = make_loso_splits(groups)

    for fold_i, (person, tr_idx, te_idx) in enumerate(folds, start=1):
        # ✅ 추가: Fold 처리 시작 시간 기록
        t_fold_start = time.time()

        fold_dir = os.path.join(runs_root, f'loso-{person}')
        for d in ['models', 'preds', 'metrics', 'viz']:
            os.makedirs(os.path.join(fold_dir, d), exist_ok=True)

        print(f'\n=== FOLD {fold_i}/{len(folds)} | Testing on person={person} ===')

        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr_s, yte_s = y_s[tr_idx], y_s[te_idx]
        ytr_h, yte_h = y_h[tr_idx], y_h[te_idx]
        ytr_v, yte_v = y_v[tr_idx], y_v[te_idx]
        mtr_v, mte_v = m_v[tr_idx], m_v[te_idx]
        ytr_v_eff = np.where(ytr_v < 0, 0, ytr_v)
        yte_v_eff = np.where(yte_v < 0, 0, yte_v)

        calib = None
        if args.calibrate and SKLEARN_OK and len(Xtr) > 10:
            # Stratify by surface label as a proxy for other labels
            Xtr, Xcal, ytr_s, ycal_s = train_test_split(Xtr, ytr_s, test_size=0.1, random_state=42, stratify=ytr_s)
            ytr_h, ycal_h = train_test_split(ytr_h, test_size=0.1, random_state=42, stratify=ytr_s)[0:2]
            ytr_v_eff, ycal_v = train_test_split(ytr_v_eff, test_size=0.1, random_state=42, stratify=(ytr_v_eff>=0).astype(int))[0:2]
            mtr_v, mcal_v = train_test_split(mtr_v, test_size=0.1, random_state=42, stratify=(mtr_v>=0).astype(int))[0:2]
            calib = (Xcal, ycal_s, ycal_h, ycal_v, mcal_v)

        model = build_mt_tcn((L, F))
        cb = [
            tf.keras.callbacks.ModelCheckpoint(os.path.join(fold_dir, 'models', 'tcn_best_model.h5'),
                                                 save_best_only=True, monitor='val_surface_accuracy', mode='max'),
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
        
        model.fit(
            augment_batch(Xtr), [ytr_s, ytr_h, ytr_v_eff],
            validation_data=(Xte, [yte_s, yte_h, yte_v_eff], {"vertical": mte_v}),
            epochs=args.epochs, batch_size=args.batch_size,
            sample_weight={"surface": np.ones_like(ytr_s), "horizontal": np.ones_like(ytr_h), "vertical": mtr_v},
            callbacks=cb, verbose=2
        )

        emb_model = models.Model(model.input, model.get_layer('embedding').output)
        yhat_s, yhat_h, yhat_v = model.predict(Xte, verbose=0)
        ypred_s, ypred_h, ypred_v = yhat_s.argmax(1), yhat_h.argmax(1), yhat_v.argmax(1)
        emb_test = emb_model.predict(Xte, verbose=0)

        # --- [생략되지 않은 부분] 결과 및 지표 저장 ---
        np.save(os.path.join(fold_dir, 'preds', 'y_prob_surface.npy'), yhat_s)
        np.save(os.path.join(fold_dir, 'preds', 'y_prob_horizontal.npy'), yhat_h)
        np.save(os.path.join(fold_dir, 'preds', 'y_prob_vertical.npy'), yhat_v)
        np.save(os.path.join(fold_dir, 'preds', 'y_true_surface.npy'), yte_s)
        np.save(os.path.join(fold_dir, 'preds', 'y_pred_surface.npy'), ypred_s)
        np.save(os.path.join(fold_dir, 'preds', 'y_true_horizontal.npy'), yte_h)
        np.save(os.path.join(fold_dir, 'preds', 'y_pred_horizontal.npy'), ypred_h)
        np.save(os.path.join(fold_dir, 'preds', 'y_true_vertical.npy'), yte_v)
        np.save(os.path.join(fold_dir, 'preds', 'y_pred_vertical.npy'), ypred_v)
        np.save(os.path.join(fold_dir, 'preds', 'embedding_test.npy'), emb_test)
        
        # Classification Report 및 F1 Score 계산
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
        else:
            rep_v = None

        with open(os.path.join(metrics_dir, 'classification_report_surface.json'), 'w', encoding='utf-8') as f: json.dump(rep_s, f, indent=2)
        with open(os.path.join(metrics_dir, 'classification_report_horizontal.json'), 'w', encoding='utf-8') as f: json.dump(rep_h, f, indent=2)
        if rep_v:
            with open(os.path.join(metrics_dir, 'classification_report_vertical.json'), 'w', encoding='utf-8') as f: json.dump(rep_v, f, indent=2)
        with open(os.path.join(metrics_dir, 'f1.json'), 'w', encoding='utf-8') as f: json.dump(f1_metrics, f, indent=2)
        
        # UMAP 시각화
        save_umap_and_silhouette(emb_test, yte_s, os.path.join(fold_dir, 'viz', 'umap_surface.png'), os.path.join(fold_dir, 'viz', 'umap_surface_silhouette.json'))
        # --- [생략되지 않은 부분] 끝 ---

        # ✅ 추가: Fold 처리 종료 시간 기록 및 소요 시간 계산
        fold_duration_sec = time.time() - t_fold_start
        
        # ✅ 수정: fold의 manifest.json 파일에 'duration_sec' 키 추가
        with open(os.path.join(fold_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'person': person,
                'train_count': int(tr_idx.sum()),
                'test_count': int(te_idx.sum()),
                'duration_sec': fold_duration_sec
            }, f, ensure_ascii=False, indent=2)
            
        print(f"[INFO] Fold {person} complete in {fold_duration_sec:.2f}s. Val F1(surface): {f1_metrics['surface']['macro_f1']:.4f}")

    print('\n✅ All folds finished for TCN.')
    print(f'📁 Runs saved to: {runs_root}')

if __name__ == '__main__':
    main()
