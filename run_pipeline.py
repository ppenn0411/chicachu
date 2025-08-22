import os
import subprocess
import time
from tqdm import tqdm

# -------------------- 실험 설정 --------------------
# 이 부분의 변수들을 수정하여 실험 범위를 조절할 수 있습니다.

# 1. 피처 추출 버전
PROJECT_TAGS = ['v1_with_eyes', 'v1_no_eyes']

# 2. 데이터셋 빌드 버전
SEQ_LENS = [15, 30, 45]
BUILD_TAG_PREFIXES = ['base', 'ema_only', 'kalman_only', 'delta_only', 'ema_kalman', 'ema_delta', 'kalman_delta', 'all_filters']

# 3. 학습 모델
MODEL_SCRIPTS = [
    'train_tcn_models.py',
    'train_lstm_models.py',
    'train_transformer_models.py',
    'train_rf_models.py'
]

# 4. 분석 스크립트
SCRIPT_ANALYZE = 'analyze_experiments.py'

# -------------------- 파이프라인 실행 함수 --------------------
def run_command(command, description=""):
    """ 주어진 명령어를 실행하고 tqdm 진행 막대를 업데이트합니다. """
    try:
        # 스크립트의 출력을 숨겨서 진행 막대가 깔끔하게 보이도록 함
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred while running: {' '.join(command)}")
        print(f"Error details: {e}")
        # 오류 발생 시 파이프라인 중단
        raise

def main():
    start_time = time.time()
    print("="*60)
    print("==                 FULL M-L PIPELINE START                ==")
    print("="*60 + "\n")

    # --- 1. 피처 추출 (preprocess_video.py) ---
    print("[PART 1] Starting Feature Extraction...")
    extraction_commands = [
        f"python preprocess_video.py --project_tag v1_with_eyes --add_eye_feats",
        f"python preprocess_video.py --project_tag v1_no_eyes"
    ]
    with tqdm(total=len(extraction_commands), desc="Extracting Features") as pbar:
        for cmd in extraction_commands:
            pbar.set_postfix_str(cmd.split('--project_tag ')[1].split(' ')[0])
            run_command(cmd)
            pbar.update(1)
    print("[PART 1] Feature Extraction Complete.\n")

    # --- 2. 데이터셋 빌드 (preprocess_sequential_data.py) ---
    print("[PART 2] Starting all Dataset Builds...")
    build_commands = []
    for p_tag in PROJECT_TAGS:
        for s_len in SEQ_LENS:
            for b_prefix in BUILD_TAG_PREFIXES:
                build_tag = f"{b_prefix}_sl{s_len}"
                ema = "--ema_alpha 0" if "ema" not in b_prefix else ""
                kalman = "--use_kalman" if "kalman" in b_prefix else ""
                delta = "--add_delta" if "delta" in b_prefix else ""
                if b_prefix == "base": ema = "--ema_alpha 0"

                cmd = f"python preprocess_sequential_data.py --project_tag {p_tag} --build_tag {build_tag} --seq_len {s_len} {ema} {kalman} {delta}"
                build_commands.append(cmd)
    
    with tqdm(total=len(build_commands), desc="Building Datasets ") as pbar:
        for cmd in build_commands:
            p_tag = cmd.split('--project_tag ')[1].split(' ')[0]
            b_tag = cmd.split('--build_tag ')[1].split(' ')[0]
            pbar.set_postfix_str(f"{p_tag}/{b_tag}")
            run_command(cmd)
            pbar.update(1)
    print("[PART 2] All Dataset Builds Complete.\n")

    # --- 3. 모델 학습 (train_..._models.py) ---
    print("[PART 3] Starting all Model Training Runs...")
    train_commands = []
    for p_tag in PROJECT_TAGS:
        for s_len in SEQ_LENS:
            for b_prefix in BUILD_TAG_PREFIXES:
                build_tag = f"{b_prefix}_sl{s_len}"
                for model_script in MODEL_SCRIPTS:
                    cmd = f"python {model_script} --project_tag {p_tag} --build_tag {build_tag}"
                    train_commands.append(cmd)

    with tqdm(total=len(train_commands), desc="Training Models   ") as pbar:
        for cmd in train_commands:
            parts = cmd.split(' ')
            model_name = parts[1].replace('train_', '').replace('_models.py', '')
            p_tag = parts[3]
            b_tag = parts[5]
            pbar.set_postfix_str(f"{p_tag}/{b_tag}/{model_name}")
            run_command(cmd)
            pbar.update(1)
    print("[PART 3] All Model Training Runs Complete.\n")

    # --- 4. 최종 분석 (analyze_experiments.py) ---
    print("[PART 4] Running Final Analysis...")
    run_command(f"python {SCRIPT_ANALYZE}")
    print("[PART 4] Analysis Complete.\n")

    end_time = time.time()
    total_duration = end_time - start_time
    print("="*60)
    print("==                 FULL PIPELINE FINISHED                 ==")
    print(f"== Total Duration: {time.strftime('%H:%M:%S', time.gmtime(total_duration))}                         ==")
    print("== Check 'experiment_summary.csv' for the final results! ==")
    print("="*60)

if __name__ == '__main__':
    main()