import os
import cv2
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

from data.dataset import DocTypeDataset
from data.augmentation import batch_transform
from utils.config_util import load_config


def load_model(model_path, num_classes, device):
    """
    모델을 로드합니다. 이 코드에서는 transformers 라이브러리의 사전 학습 모델을 사용합니다.
    """
    model = AutoModelForImageClassification.from_pretrained(
        "amaye15/SwinV2-Base-Document-Classifier",
        ignore_mismatched_sizes=True,
        num_labels=num_classes
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def inference(model, dataloader, device):
    """
    모델을 사용하여 추론을 수행합니다.
    """
    ids_list = []
    preds_list = []
    with torch.no_grad():
        for ids, images, _ in tqdm(dataloader, desc="Inference", leave=False):
            images = images.to(device)
            outputs = model(images).logits
            preds = outputs.argmax(dim=1).detach().cpu().numpy()

            ids_list.extend(ids)
            preds_list.extend(preds)

    return ids_list, preds_list


def save_predictions(ids, preds, classes, cfg):
    """
    예측 결과를 저장합니다. 이미지 파일을 분류하여 저장하고, 결과를 CSV 파일로 저장합니다.
    """
    kr_classes = {
        "account_number": "계좌번호",
        "application_for_payment_of_pregnancy_medical_expenses": "임신_의료비_지급_신청서",
        "car_dashboard": "자동차_대시보드",
        "confirmation_of_admission_and_discharge": "입퇴원_확인서",
        "diagnosis": "진단서",
        "driver_lisence": "운전면허증",
        "medical_bill_receipts": "의료비_영수증",
        "medical_outpatient_certificate": "외래_진료_증명서",
        "national_id_card": "주민등록증",
        "passport": "여권",
        "payment_confirmation": "결제_확인서",
        "pharmaceutical_receipt": "약국_영수증",
        "prescription": "처방전",
        "resume": "이력서",
        "statement_of_opinion": "의견진술서",
        "vehicle_registration_certificate": "차량_등록_증명서",
        "vehicle_registration_plate": "차량 등록 번호판"
    }

    pred_dir = f"{cfg['saved_dir']}/preds"
    os.makedirs(pred_dir, exist_ok=True)

    # 클래스별 폴더 생성
    for c in classes:
        kor_c = kr_classes[c]
        os.makedirs(f"{pred_dir}/{kor_c}", exist_ok=True)

    # 예측 결과를 폴더에 저장
    for id, pred in zip(ids, preds):
        image = cv2.imread(f"{cfg['test_img_path']}/{id}")
        en_str_label = classes[pred]
        ko_str_label = kr_classes[en_str_label]
        cv2.imwrite(f"{pred_dir}/{ko_str_label}/{id}", image)

    # 결과를 CSV로 저장
    df = pd.DataFrame(list(zip(ids, preds)), columns=['ID', 'target'])
    df.to_csv(f"{pred_dir}/submission.csv", index=False)


def main(cfg):
    """
    테스트 데이터에 대한 추론을 수행하고 결과를 저장합니다.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = batch_transform(cfg['img_h'], cfg['img_w'])
    test_dataset = DocTypeDataset(cfg['test_img_path'], cfg['test_csv_path'], cfg['meta_path'], transform, cfg["one_hot_encoding"])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    # 학습된 모델 로드
    model = load_model(f"{cfg['saved_dir']}/weights/best.pth", test_dataset.num_classes, device)
    
    # 추론
    ids, preds = inference(model, test_dataloader, device)

    # 예측 결과 저장
    save_predictions(ids, preds, test_dataset.classes, cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--saved_dir', type=str, default='./runs/2024-07-31-16-30-20', help='Path to Trained Dir')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    dir_path = args.saved_dir
    cfg = load_config(f"{dir_path}/config.yaml")
    cfg['saved_dir'] = dir_path

    main(cfg)
