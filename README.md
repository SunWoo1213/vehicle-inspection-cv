# 차량 외관 검사: 차종 분류 + 부품 탐지 + 부품 훼손 탐지

컴퓨터비전 미니프로젝트 #1의 참조 코드 저장소입니다. 차량 사진 한 장에서 **① 차종 분류**, **② 차량 부품 탐지**, **③ 부품 훼손 탐지**를 수행하는 것이 목표입니다. 세 기능은 아래 기존 코드가 각각 따로 구현하고 있으며, 이번 프로젝트에서는 이를 하나의 파이프라인으로 연결하고 부품별 훼손 리포트를 추가로 구현합니다.

| 단계 | 폴더 | 모델 | 출력 | 데이터셋 |
| --- | --- | --- | --- | --- |
| ① 차종 분류 | `vehicle_classification/` | TransFG (ViT-B/16) | 차종 108클래스 | AI Hub 091. 차량외관 영상 데이터 |
| ② 부품 탐지 | `part_detection/` | YOLOv5n | 부품 15클래스 바운딩 박스 | AI Hub 091. 차량외관 영상 데이터 |
| ③ 훼손 탐지 | `damage_segmentation/` | U-Net (ResNet34) | 파손 4종 픽셀 마스크 | AI Hub 차량 파손 이미지 |

데이터, 사전학습 가중치, 학습 결과는 저장소에 포함하지 않았습니다. 데이터는 [AI Hub](https://www.aihub.or.kr)에서 신청 후 내려받아야 합니다.

## 계획한 통합 파이프라인

```
차량 사진 ─▶ ② YOLOv5: car 박스 + 부품 박스
              ├─▶ car 박스로 자른 이미지 ─▶ ① TransFG: 차종
              └─▶ ③ U-Net: 파손 마스크(Scratched/Separated/Crushed/Breakage)
                         └─▶ 부품 박스와 겹쳐 부품별 훼손 종류·면적 리포트
```

## ① 차종 분류 — `vehicle_classification/`

[TransFG](https://github.com/TACJu/TransFG) (MIT) 코드를 차종 분류용으로 수정한 버전입니다.

- `FILE_LOAD` 데이터셋(108클래스) 추가: `utils/dataset.py`의 `FileLoad`가 라벨의 `P00.차량전체` bbox로 차량만 잘라 분류하고, 데이터를 청크 단위로 나눠 읽습니다.
- `converter_normal_car.py`: AI Hub JSON → `images.txt`, `image_class_label_bboxes.txt`, `train_val_test_split.txt` (8:1:1)
- 입력 448×448, SGD lr 3e-2, cosine decay, 10,000 step, Apex fp16

```bash
bash setup.sh   # NVIDIA Apex 설치
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
python main.py --name run1 --dataset FILE_LOAD --data_root <데이터 경로> \
  --pretrained_dir ViT-B_16.npz --split overlap --fp16 --run-mode train   # 테스트: --run-mode test
```

## ② 부품 탐지 — `part_detection/`

[Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) (GPL-3.0)에 차량 부품 데이터 설정을 추가했습니다.

- 클래스 15개: car, Front bumper, Rear bumper, Wheel, A Pillar, C Pillar, Side mirror, Front door, Back door, Radiator grill, Head lamp, Rear lamp, Bonnet, Trunk, Roof (`yolov5/data/CarpartDetection.yaml`)
- 원본 학습 결과(yolov5n, 640px, 150 epoch): 검증 이미지 32,385장에서 Precision 0.939, Recall 0.925, mAP@0.5 0.962, mAP@0.5:0.95 0.782

```bash
cd part_detection/prepare_dataset
python prepare_yolo.py --output_yolo_data <YOLO 데이터 저장 경로>   # --test_code True: 10,000장 샘플
cd ../yolov5
python train.py --data data/CarpartDetection.yaml --weights yolov5n.pt --img 640 --epochs 150 --device 0
python val.py --data data/CarpartDetection.yaml --weights runs/train/exp/weights/best.pt --task test --device 0
```

## ③ 부품 훼손 탐지 — `damage_segmentation/`

[segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) U-Net(ResNet34, ImageNet 사전학습)으로 파손 영역을 분할합니다.

- 파손: Scratched, Separated, Crushed, Breakage 라벨마다 이진 U-Net 1개씩 학습
- 부위: 16클래스 다중 클래스 U-Net
- 입력 256×256, COCO 형식 라벨, 평가 지표 mIoU

```bash
cd damage_segmentation
docker build -t damage-seg .   # 또는 pip install -r requirements.txt (+ pycocotools)
python main.py --train y --task damage --label all
python main.py --eval y --task damage --dataset test
```

데이터와 가중치는 `../data/` 아래(`datainfo/`, `Dataset/1.원천데이터/`, `weight/`, `result_log/`)에 있다고 가정합니다.

## 출처 및 이용 안내

이 저장소의 코드는 [AI Hub](https://www.aihub.or.kr)에서 「차량외관 영상 데이터」, 「차량 파손 이미지」 데이터셋과 함께 제공된 소스코드를 바탕으로 합니다. 수업(컴퓨터비전 미니프로젝트 #1) 과제 제출을 위한 **교육·비상업 목적**으로만 올렸습니다.

- 원 코드의 저작권은 각 저작자와 AI Hub 데이터 구축 기관에 있습니다.
- 데이터, 이미지, 라벨, 학습 가중치는 포함하지 않았습니다. 데이터는 AI Hub에서 이용 신청 후 받아야 하며, AI Hub 이용약관을 따릅니다.
- 권리자의 요청이 있으면 해당 코드를 즉시 삭제하겠습니다.

| 구성 요소 | 원 출처 | 라이선스 |
| --- | --- | --- |
| `vehicle_classification/` | [TransFG](https://github.com/TACJu/TransFG) — He et al., *TransFG: A Transformer Architecture for Fine-grained Recognition*, 2021 (AI Hub 차종 분류용 수정본) | MIT (`vehicle_classification/LICENSE`) |
| `part_detection/yolov5/` | [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) (AI Hub 부품 탐지 설정 추가) | GPL-3.0 (`part_detection/yolov5/LICENSE`) |
| `part_detection/prepare_dataset/` | AI Hub 「차량외관 영상 데이터」 제공 코드 | 별도 표기 없음 |
| `damage_segmentation/` | AI Hub 「차량 파손 이미지」 제공 코드, [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) 사용 | 별도 표기 없음 |
