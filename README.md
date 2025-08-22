# 칫솔질 동작 분석 AI 파이프라인

**영상 데이터로부터 특징을 추출하고, 다양한 머신러닝/딥러닝 모델을 체계적으로 학습 및 평가하는 MLOps 파이프라인**

이 프로젝트는 원본 비디오(.mp4)로부터 MediaPipe를 사용하여 손과 얼굴의 랜드마크를 추출하고, 이를 시계열 특징으로 가공합니다. 이후 TCN, LSTM, Transformer, Random Forest 등 4가지 모델 아키텍처와 다양한 데이터 전처리 조합을 체계적으로 실험하여 최적의 동작 분류 모델을 찾는 것을 목표로 합니다. 모든 과정은 스크립트를 통해 자동화되어 있으며, 재현 가능한 실험을 보장합니다.

## 1. 프로젝트 주요 특징

* **고품질 피처 엔지니어링**: MediaPipe를 활용한 랜드마크 추출, 코(Nose) 기준 좌표 정규화, 안정적인 눈(Eye) 기준점 추가.
* **강건한 데이터 전처리**:
    * **레터박스(Letterbox)**: 영상 비율을 유지하여 데이터 왜곡 방지.
    * **시간적 평균화(Temporal Averaging)**: 정보 손실을 최소화하는 FPS 리샘플링.
    * **노이즈 제거**: 얼굴/손 미검출 프레임에 대한 체계적인 처리 (무시 또는 Zero-fill).
* **체계적인 모델 평가**:
    * **LOSO(Leave-One-Subject-Out) 교차 검증**: 사용자별 일반화 성능을 엄격하게 측정.
    * **다양한 모델 아키텍처 비교**: TCN, LSTM, Transformer, Random Forest 모델 동시 실험.
* **완전 자동화된 파이프라인**: 단일 스크립트 실행으로 피처 추출부터 데이터셋 빌드, 모델 학습, 최종 분석까지 모든 과정 자동화.
* **상세한 결과 분석**: 모든 실험 조합(192개)에 대한 성능, 안정성(표준편차), 효율성(학습 시간)을 종합 분석하여 최적의 조합을 과학적으로 선정.

---
## 2. 폴더 구조

프로젝트의 모든 스크립트를 실행한 후의 최종 폴더 구조입니다.

PROJECT_ROOT/
│
├── video_data/                  <-- 1. 원본 영상 저장소
│   ├── 왼쪽-협측/
│   │   └── P01/
│   │       └── clip_01.mp4
│   └── ...
│
├── v1_with_eyes/                <-- 2. "눈 피처 사용" 버전의 모든 산출물
│   ├── data_v1_with_eyes/         (1차 정제된 피처 .npy 파일들)
│   └── artifacts_v1_with_eyes/    (2차 가공된 학습용 데이터셋 - 24개 빌드)
│
├── v1_no_eyes/                  <-- 2. "눈 피처 제외" 버전의 모든 산출물
│   ├── data_v1_no_eyes/
│   └── artifacts_v1_no_eyes/
│
├── logs/                        <-- 3. 피처 추출 과정 상세 로그
│   ├── v1_with_eyes_extraction_log.json
│   └── v1_no_eyes_extraction_log.json
│
├── runs/                        <-- 4. 모델 학습 결과 (192개 실험 결과)
│   └── v1_with_eyes/
│       └── base_sl15/
│           ├── tcn_20250822_180000/
│           │   └── loso-P01/
│           │       ├── models/
│           │       ├── metrics/
│           │       └── ...
│           └── ... (다른 모델 결과)
│
├── experiment_summary.csv       <-- ⭐ 최종 종합 분석표
├── top1_cm_surface.jpg          <-- ⭐ 최고 모델 실패 사례 분석 (혼동 행렬)
│
├── preprocess_video.py
├── preprocess_sequential_data.py
├── train_tcn_models.py
├── train_lstm_models.py
├── train_transformer_models.py
├── train_rf_models.py
├── analyze_experiments.py
├── run_full_pipeline.bat        <-- (Windows용 마스터 실행 스크립트)
└── requirements.txt


---
## 3. 환경 설정 및 설치

본 프로젝트는 **Anaconda** 환경과 **Python 3.10** 버전을 기준으로 작성되었습니다.

1.  **Anaconda 가상환경 생성 및 활성화**
    ```bash
    conda create -n toothbrush python=3.10
    conda activate toothbrush
    ```

2.  **필수 라이브러리 설치**
    프로젝트 폴더에 포함된 `requirements.txt` 파일을 사용하여 모든 라이브러리를 한 번에 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

3.  **GPU 사용 설정 (NVIDIA)**
    딥러닝 모델 학습 시 GPU를 사용하려면, 위 라이브러리 설치 전에 **NVIDIA 그래픽 드라이버, CUDA, cuDNN**이 시스템에 설치되어 있어야 합니다.

---
## 4. 전체 파이프라인 실행 방법

### 1단계: 원본 비디오 준비

`video_data` 폴더를 생성하고, 그 안에 아래와 같은 구조로 원본 비디오 파일(.mp4 등)을配置합니다.
`video_data/<라벨명>/<사람ID>/<클립이름>.mp4`

### 2단계: 마스터 파이프라인 스크립트 실행

모든 과정은 `run_full_pipeline.bat` (또는 `run_pipeline.py`) 스크립트 하나로 자동화됩니다. 터미널에서 아래 명령어를 실행하세요.

```bash
# Windows
run_full_pipeline.bat
```

⚠️ 경고: 이 스크립트는 192개의 모델을 학습시키는 매우 긴 작업입니다. 컴퓨터 사양에 따라 수십 시간에서 며칠이 소요될 수 있습니다.

### 3단계: 최종 결과 분석

모든 학습이 완료된 후, 아래 스크립트를 실행하여 모든 실험 결과를 취합하고 종합 분석표를 생성합니다. (run_full_pipeline 스크립트에 마지막 단계로 포함되어 있어 자동으로 실행됩니다.)

```bash
python analyze_experiments.py
```

---
## 5. 스크립트 설명

| 파일 이름                  | 역할 |
|----------------------------|------------------------------------------------|
| preprocess_video.py        | 1단계: 원본 영상(.mp4)에서 랜드마크를 추출하여 피처(.npy)를 생성합니다. |
| preprocess_sequential_data.py | 2단계: 피처를 입력받아 필터링 및 윈도잉하여 최종 학습용 데이터셋을 빌드합니다. |
| train_tcn_models.py        | 3단계: TCN 모델을 학습하고 평가 결과를 저장합니다. |
| train_lstm_models.py       | 3단계: LSTM 모델을 학습하고 평가 결과를 저장합니다. |
| train_transformer_models.py| 3단계: Transformer 모델을 학습하고 평가 결과를 저장합니다. |
| train_rf_models.py         | 3단계: Random Forest 모델을 학습하고 평가 결과를 저장합니다. |
| analyze_experiments.py     | 4단계: 모든 runs 폴더의 결과를 취합하여 종합 분석표와 시각화 자료를 생성합니다. |
| run_full_pipeline.bat      | 마스터 스크립트: 위 모든 과정을 순서대로 자동 실행합니다. |

---
## 6. 최종 결과물 해석

- **experiment_summary.csv**: 모든 실험 결과를 담은 최종 요약표입니다. 각 조합의 **평균 성능(mean), 성능 안정성(std), 학습 속도(duration)**를 비교하여 최적의 모델을 선정할 수 있습니다.
- **top1_cm_*.jpg**: 종합 순위 1위 모델의 **혼동 행렬(Confusion Matrix)**입니다. 모델이 어떤 라벨을 서로 헷갈리는지 시각적으로 분석할 수 있습니다.
- **runs/** 폴더: 모든 개별 실험의 상세 결과(학습된 모델, 예측값, 평가지표 등)가 보관된 아카이브입니다.
