import os
import cv2
import timm
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import DocTypeDataset
from data.augmentation import batch_transform

from utils.config_util import load_config


def load_model(model_path, model_name, num_classes, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    return model


def inference(model, dataloader, device):
    model.eval()
    ids_list = []
    preds_list = []
    with torch.no_grad():
        for id, images in tqdm(dataloader):
            images = images.to(device)
            preds = model(images)

            ids_list.extend(id)
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())

    return ids_list, preds_list


def save_predictions(ids, preds, classes, cfg):
    kr_classes = {"account_number" : "계좌번호",
                  "application_for_payment_of_pregnancy_medical_expenses" : "임신_의료비_지급_신청서",
                  "car_dashboard" : "자동차_대시보드",
                  "confirmation_of_admission_and_discharge" : "입퇴원_확인서",
                  "diagnosis" : "진단서",
                  "driver_lisence" : "운전면허증",
                  "medical_bill_receipts" : "의료비_영수증",
                  "medical_outpatient_certificate" : "외래_진료_증명서",
                  "national_id_card" : "주민등록증",
                  "passport" : "여권",
                  "payment_confirmation" : "결제_확인서",
                  "pharmaceutical_receipt" : "약국_영수증",
                  "prescription" : '처방전',
                  "resume" : "이력서",
                  "statement_of_opinion" : "의견진술서",
                  "vehicle_registration_certificate" : "차량_등록_증명서",
                  "vehicle_registration_plate" : "차량 등록 번호판"}
    
    pred_dir = f"{cfg['saved_dir']}/preds"
    for c in classes:
        kor_c = kr_classes[c]
        os.makedirs(f"{pred_dir}/{kor_c}", exist_ok=True)
    
    for id, pred in zip(ids, preds):
        image = cv2.imread(f"{cfg['data_path']}/test/{id}")
        en_str_label = classes[pred]
        ko_str_label = kr_classes[en_str_label]
        cv2.imwrite(f"{pred_dir}/{ko_str_label}/{id}", image)

    df = pd.DataFrame(list(zip(ids, preds)), columns=['ID', 'target'])
    df.to_csv(f"{pred_dir}/submission.csv", index=False)


def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = batch_transform(cfg['img_h'], cfg['img_w'])
    test_dataset = DocTypeDataset(cfg['test_img_path'], cfg['test_csv_path'], cfg['meta_path'], transform, cfg["one_hot_encoding"])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    
    model = load_model(f"{cfg['saved_dir']}/weights/best.pth", cfg['model_name'], test_dataset.num_classes, device)
    ids, preds = inference(model, test_dataloader, device)

    file_name = cfg['saved_dir'].split('/')[-1]
    print(file_name)
    output_file = f"{cfg['saved_dir']}/{file_name}.csv"
    print(output_file)
    save_predictions(ids, preds, output_file)


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