import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ----- 설정 -----
RUNS_ROOT = "runs"
OUTPUT_CSV = "experiment_summary.csv"
TOP_N_MODELS_TO_ANALYZE = 1 # 상세 분석(Confusion Matrix)을 생성할 상위 모델 개수

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    print("Malgun Gothic 폰트가 설치되어 있지 않아 한글이 깨질 수 있습니다.")

def plot_confusion_matrix(y_true, y_pred, class_names, title, output_path):
    """ Confusion Matrix를 그리고 이미지 파일로 저장합니다. """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('실제 라벨 (True Label)')
    plt.xlabel('예측 라벨 (Predicted Label)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Confusion Matrix saved to: {output_path}")

def analyze_all_results():
    """ 'runs' 폴더를 탐색하여 모든 실험 결과를 취합하고 종합 분석을 수행합니다. """
    all_results = []
    
    # 분석에 필요한 manifest.json과 f1.json을 모두 찾습니다.
    manifest_files = glob.glob(os.path.join(RUNS_ROOT, '**', 'loso-*', 'manifest.json'), recursive=True)

    for manifest_path in manifest_files:
        try:
            fold_dir = os.path.dirname(manifest_path)
            path_parts = fold_dir.split(os.sep)
            
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            f1_path = os.path.join(fold_dir, 'metrics', 'f1.json')
            if not os.path.exists(f1_path): continue
            with open(f1_path, 'r', encoding='utf-8') as f:
                f1_data = json.load(f)

            result = {
                'project_tag': path_parts[1],
                'build_tag': path_parts[2],
                'model_info': path_parts[3],
                'model_type': path_parts[3].split('_')[0],
                'test_person': manifest_data['person'],
                'duration_sec': manifest_data.get('duration_sec'),
                'f1_surface': f1_data.get('surface', {}).get('macro_f1'),
                'f1_horizontal': f1_data.get('horizontal', {}).get('macro_f1'),
                'f1_vertical': f1_data.get('vertical', {}).get('macro_f1'),
            }
            all_results.append(result)
        except Exception as e:
            print(f"Warning: Could not process folder {os.path.dirname(manifest_path)}. Error: {e}")

    if not all_results:
        print("처리할 결과 파일을 찾지 못했습니다. 학습을 먼저 실행해주세요.")
        return

    df = pd.DataFrame(all_results)
    
    # 1. 자원 효율성 및 성능 안정성 분석
    agg_funcs = {
        'f1_surface': ['mean', 'std'],
        'f1_horizontal': ['mean', 'std'],
        'f1_vertical': ['mean', 'std'],
        'duration_sec': ['mean']
    }
    summary_df = df.groupby(['project_tag', 'build_tag', 'model_type']).agg(agg_funcs).round(4)
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.sort_values(by='f1_surface_mean', ascending=False)
    
    summary_df.to_csv(OUTPUT_CSV, encoding='utf-8-sig')
    print(f"\n✅ 종합 분석 결과가 '{OUTPUT_CSV}' 파일로 저장되었습니다.")
    print("\n--- 상위 5개 모델 조합 (성능, 안정성, 효율성 종합 분석) ---")
    print(summary_df.head(5))

    # 2. 실패 사례 분석 (Confusion Matrix 생성)
    print("\n--- 상위 모델 실패 사례 분석 (Confusion Matrix 생성) ---")
    
    for i in range(min(TOP_N_MODELS_TO_ANALYZE, len(summary_df))):
        project_tag, build_tag, model_type = summary_df.index[i]
        model_info_prefix = model_type
        
        print(f"\nAnalyzing Top-{i+1} model: {project_tag}/{build_tag}/{model_type}")
        
        # 메타 정보 로딩 (라벨 이름 확인용)
        meta_path = os.path.join('artifacts', project_tag, build_tag, f'encoder_meta_{build_tag}.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # 해당 실험의 모든 fold에서 y_true, y_pred 불러오기
        all_true_s, all_pred_s, all_true_h, all_pred_h, all_true_v, all_pred_v = [[] for _ in range(6)]
        pred_folders = glob.glob(os.path.join(RUNS_ROOT, project_tag, build_tag, f'{model_info_prefix}_*', 'loso-*', 'preds'))
        
        for pred_folder in pred_folders:
            all_true_s.append(np.load(os.path.join(pred_folder, 'y_true_surface.npy')))
            all_pred_s.append(np.load(os.path.join(pred_folder, 'y_pred_surface.npy')))
            all_true_h.append(np.load(os.path.join(pred_folder, 'y_true_horizontal.npy')))
            all_pred_h.append(np.load(os.path.join(pred_folder, 'y_pred_horizontal.npy')))
            
            true_v, pred_v = np.load(os.path.join(pred_folder, 'y_true_vertical.npy')), np.load(os.path.join(pred_folder, 'y_pred_vertical.npy'))
            mask_v = true_v != -1
            all_true_v.append(true_v[mask_v])
            all_pred_v.append(pred_v[mask_v])

        # Confusion Matrix 생성 및 저장
        plot_confusion_matrix(np.concatenate(all_true_s), np.concatenate(all_pred_s), meta['SURFACES'],
                              f'Top-{i+1} Model CM - Surface\n({project_tag}/{build_tag}/{model_type})', f'top{i+1}_cm_surface.jpg')
        plot_confusion_matrix(np.concatenate(all_true_h), np.concatenate(all_pred_h), meta['HORIZONTALS'],
                              f'Top-{i+1} Model CM - Horizontal\n({project_tag}/{build_tag}/{model_type})', f'top{i+1}_cm_horizontal.jpg')
        if np.concatenate(all_true_v).size > 0:
            plot_confusion_matrix(np.concatenate(all_true_v), np.concatenate(all_pred_v), meta['VERTICALS'],
                                  f'Top-{i+1} Model CM - Vertical\n({project_tag}/{build_tag}/{model_type})', f'top{i+1}_cm_vertical.jpg')

if __name__ == '__main__':
    analyze_all_results()