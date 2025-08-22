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
    m_s = np.load(os.path.join(r, f'mask_surface_mt_{tag}.npy'))
    m_h = np.load(os.path.join(r, f'mask_horizontal_mt_{tag}.npy'))
    m_v = np.load(os.path.join(r, f'mask_vertical_mt_{tag}.npy'))
    groups = np.load(os.path.join(r, f'groups_mt_{tag}.npy'))
    meta = json.load(open(os.path.join(r, f'encoder_meta_{tag}.json'),'r',encoding='utf-8'))
    return X, y_s, y_h, y_v, m_s, m_h, m_v, groups, meta

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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v1', help='버전 태그 (예: v1, v2)')
    ap.add_argument('--artifacts_root', default=None, help='입력 배열 루트. 생략 시 artifacts/<tag>')
    ap.add_argument('--cv', default='loso', choices=['loso'])
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--augment', action='store_true', default=False)
    ap.add_argument('--calibrate', action='store_true', default=True,
                    help='온도 스케일링으로 calibration 수행(학습셋의 10% 사용)')
    ap.add_argument('--umap_neighbors', type=int, default=20)
    ap.add_argument('--umap_min_dist', type=float, default=0.2)
    args = ap.parse_args()

    artifacts_root = args.artifacts_root or os.path.join('artifacts', args.tag)
    if not os.path.isdir(artifacts_root):
        raise FileNotFoundError(f"입력 배열 폴더가 없습니다: {artifacts_root}")

    X, y_s, y_h, y_v, m_s, m_h, m_v, groups, meta = load_arrays(args.tag, artifacts_root)
    N, L, F = X.shape

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    runs_root = os.path.join('runs', args.tag, timestamp)
    os.makedirs(runs_root, exist_ok=True)

    def augment_batch(Xb):
        if not args.augment: return Xb
        noise = np.random.normal(scale=0.01, size=Xb.shape).astype(np.float32)
        return Xb + noise

    folds = make_loso_splits(groups)

    with open(os.path.join(runs_root, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'tag': args.tag, 'timestamp': timestamp,
            'seq_len': int(L), 'feat_dim': int(F),
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'augment': bool(args.augment), 'calibrate': bool(args.calibrate),
            'umap_neighbors': args.umap_neighbors, 'umap_min_dist': args.umap_min_dist,
            'artifacts_root': os.path.abspath(artifacts_root)
        }, f, ensure_ascii=False, indent=2)

    for fold_i, (person, tr_idx, te_idx) in enumerate(folds, start=1):
        fold_dir = os.path.join(runs_root, f'loso-{person}')
        os.makedirs(os.path.join(fold_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'preds'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'viz'), exist_ok=True)

        print(f'=== FOLD {fold_i}/{len(folds)} | person={person} ===')

        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr_s, yte_s = y_s[tr_idx], y_s[te_idx]
        ytr_h, yte_h = y_h[tr_idx], y_h[te_idx]
        ytr_v, yte_v = y_v[tr_idx], y_v[te_idx]
        mtr_v, mte_v = m_v[tr_idx], m_v[te_idx]

        ytr_v_eff = np.where(ytr_v < 0, 0, ytr_v)
        yte_v_eff = np.where(yte_v < 0, 0, yte_v)

        if args.calibrate and SKLEARN_OK and len(Xtr) > 10:
            Xtr, Xcal, ytr_s2, ycal_s = train_test_split(Xtr, ytr_s, test_size=0.1, random_state=42, stratify=ytr_s)
            ytr_h2, ycal_h = train_test_split(ytr_h, test_size=0.1, random_state=42, stratify=ytr_s)[0:2]
            ytr_v2, ycal_v = train_test_split(ytr_v_eff, test_size=0.1, random_state=42, stratify=(ytr_v_eff>=0).astype(int))[0:2]
            mtr_v2, mcal_v = train_test_split(mtr_v, test_size=0.1, random_state=42, stratify=(ytr_v>=0).astype(int))[0:2]
            ytr_s, ytr_h, ytr_v_eff, mtr_v = ytr_s2, ytr_h2, ytr_v2, mtr_v2
            calib = (Xcal, ycal_s, ycal_h, ycal_v, mcal_v)
        else:
            calib = None

        model = build_mt_tcn((L, F))
        cb = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(fold_dir, 'models', 'tcn_model_all.h5'),
                save_best_only=True, monitor='val_surface_accuracy', mode='max'),
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
        ]

        model.fit(
            augment_batch(Xtr), [ytr_s, ytr_h, ytr_v_eff],
            validation_data=(Xte, [yte_s, yte_h, yte_v_eff], {"vertical": mte_v}),
            epochs=args.epochs, batch_size=args.batch_size,
            sample_weight={"surface": np.ones_like(ytr_s),
                           "horizontal": np.ones_like(ytr_h),
                           "vertical": mtr_v},
            callbacks=cb, verbose=2
        )

        emb_layer = model.get_layer('embedding').output
        emb_model = tf.keras.Model(model.input, emb_layer)
        emb_model.save(os.path.join(fold_dir, 'models', 'tcn_embed.h5'))

        yhat_s, yhat_h, yhat_v = model.predict(Xte, verbose=0)
        np.save(os.path.join(fold_dir, 'preds', 'y_prob_surface.npy'), yhat_s)
        np.save(os.path.join(fold_dir, 'preds', 'y_prob_horizontal.npy'), yhat_h)
        np.save(os.path.join(fold_dir, 'preds', 'y_prob_vertical.npy'), yhat_v)

        ypred_s = yhat_s.argmax(1); ypred_h = yhat_h.argmax(1); ypred_v = yhat_v.argmax(1)
        np.save(os.path.join(fold_dir, 'preds', 'y_true_surface.npy'), yte_s)
        np.save(os.path.join(fold_dir, 'preds', 'y_pred_surface.npy'), ypred_s)
        np.save(os.path.join(fold_dir, 'preds', 'y_true_horizontal.npy'), yte_h)
        np.save(os.path.join(fold_dir, 'preds', 'y_pred_horizontal.npy'), ypred_h)
        np.save(os.path.join(fold_dir, 'preds', 'y_true_vertical.npy'), yte_v)
        np.save(os.path.join(fold_dir, 'preds', 'y_pred_vertical.npy'), ypred_v)

        emb_test = emb_model.predict(Xte, verbose=0)
        np.save(os.path.join(fold_dir, 'preds', 'embedding_test.npy'), emb_test)

        # ===== 성능 지표 =====
        def f1_macro_min(y_true, y_pred, n_classes):
            if not SKLEARN_OK: return None, None
            macro = float(f1_score(y_true, y_pred, average='macro', labels=list(range(n_classes))))
            rep = None
            try:
                from sklearn.metrics import classification_report
                rep = classification_report(y_true, y_pred, labels=list(range(n_classes)),
                                            output_dict=True, zero_division=0)
                per_class_f1 = [rep[str(i)]['f1-score'] for i in range(n_classes)]
                min_f1 = float(np.min(per_class_f1)) if per_class_f1 else None
            except Exception:
                min_f1 = None
            return macro, min_f1, rep

        metrics_dir = os.path.join(fold_dir, 'metrics')

        f1_s_macro, f1_s_min, rep_s = f1_macro_min(yte_s, ypred_s, 3)
        f1_h_macro, f1_h_min, rep_h = f1_macro_min(yte_h, ypred_h, 3)
        if SKLEARN_OK and ( (yte_v>=0).sum() > 0 ):
            mask = (yte_v >= 0)
            f1_v_macro, f1_v_min, rep_v = f1_macro_min(yte_v[mask], ypred_v[mask], 2)
        else:
            f1_v_macro, f1_v_min, rep_v = None, None, None

        if rep_s is not None:
            json.dump(rep_s, open(os.path.join(metrics_dir,'classification_report_surface.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)
        if rep_h is not None:
            json.dump(rep_h, open(os.path.join(metrics_dir,'classification_report_horizontal.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)
        if rep_v is not None:
            json.dump(rep_v, open(os.path.join(metrics_dir,'classification_report_vertical.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)

        json.dump({
            'surface': {'macro_f1': f1_s_macro, 'min_f1': f1_s_min},
            'horizontal': {'macro_f1': f1_h_macro, 'min_f1': f1_h_min},
            'vertical': {'macro_f1': f1_v_macro, 'min_f1': f1_v_min}
        }, open(os.path.join(metrics_dir,'f1.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)

        # ===== Calibration (ECE + Temp Scaling) =====
        ece_dict, temp_dict = {}, {}
        def probs_to_logits(p):
            p = np.clip(p, 1e-8, 1-1e-8)
            return np.log(p)

        prob_s_test = yhat_s
        ece_raw = ece_score(prob_s_test, yte_s, n_bins=15)
        ece_dict['surface_raw'] = ece_raw

        if args.calibrate:
            if 'calib' not in locals() or calib is None:
                idx = np.random.RandomState(42).choice(len(Xtr), max(10, len(Xtr)//10), replace=False)
                Xcal, ycal_s = Xtr[idx], ytr_s[idx]
                prob_s_cal = model.predict(Xcal, verbose=0)[0]
            else:
                Xcal, ycal_s = calib[0], calib[1]
                prob_s_cal = model.predict(Xcal, verbose=0)[0]

            logits_cal = probs_to_logits(prob_s_cal)
            best_T, best_nll = find_best_temperature(logits_cal, ycal_s, (0.5, 5.0, 51))
            temp_dict['surface_T'] = best_T
            temp_dict['surface_nll'] = best_nll

            logits_test = probs_to_logits(prob_s_test)
            prob_s_test_cal = softmax(temperature_scale_logits(logits_test, best_T))
            ece_cal = ece_score(prob_s_test_cal, yte_s, n_bins=15)
            ece_dict['surface_temp_scaled'] = ece_cal
            np.save(os.path.join(fold_dir, 'preds', 'y_prob_surface_calibrated.npy'), prob_s_test_cal)

        json.dump(ece_dict, open(os.path.join(metrics_dir,'ece.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)
        if temp_dict:
            json.dump(temp_dict, open(os.path.join(metrics_dir,'temperature.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)

        # ===== UMAP & Silhouette =====
        if UMAP_OK and SKLEARN_OK:
            save_umap_and_silhouette(
                emb_test, yte_s,
                os.path.join(fold_dir, 'viz', 'umap_surface.png'),
                os.path.join(fold_dir, 'viz', 'umap_surface_silhouette.json'),
            )
            save_umap_and_silhouette(
                emb_test, yte_h,
                os.path.join(fold_dir, 'viz', 'umap_horizontal.png'),
                os.path.join(fold_dir, 'viz', 'umap_horizontal_silhouette.json'),
            )
            if (yte_v>=0).sum() > 0:
                mask = (yte_v >= 0)
                save_umap_and_silhouette(
                    emb_test[mask], yte_v[mask],
                    os.path.join(fold_dir, 'viz', 'umap_vertical.png'),
                    os.path.join(fold_dir, 'viz', 'umap_vertical_silhouette.json'),
                )

        with open(os.path.join(fold_dir, 'metrics', 'summary.txt'), 'w', encoding='utf-8') as f:
            f.write(f'person={person}\n')
            f.write(f'F1(surface): macro={f1_s_macro}, min={f1_s_min}\n')
            f.write(f'F1(horizontal): macro={f1_h_macro}, min={f1_h_min}\n')
            f.write(f'F1(vertical): macro={f1_v_macro}, min={f1_v_min}\n')
            f.write(f'ECE(surface raw)={ece_dict.get("surface_raw")}\n')
            if 'surface_temp_scaled' in ece_dict:
                f.write(f'ECE(surface temp)={ece_dict.get("surface_temp_scaled")}\n')

        json.dump({
            'person': person,
            'train_count': int(tr_idx.sum()),
            'test_count': int(te_idx.sum())
        }, open(os.path.join(fold_dir, 'manifest.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print('✅ All folds finished.')
    print('📁 Runs saved to:', runs_root)

if __name__ == '__main__':
    main()
