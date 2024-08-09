import os
import cv2
import timm
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.nn import functional as F

from data.augmentation import batch_transform, test_transform

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
        for id, images, _ in tqdm(dataloader):
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
        image = cv2.imread(f"{cfg['test_img_path']}/{id}")
        en_str_label = classes[pred]
        ko_str_label = kr_classes[en_str_label]
        cv2.imwrite(f"{pred_dir}/{ko_str_label}/{id}", image)

    df = pd.DataFrame(list(zip(ids, preds)), columns=['ID', 'target'])
    df.to_csv(f"{pred_dir}/submission.csv", index=False)


def augment_image(image):
    rotated_images = [
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 
        cv2.rotate(image, cv2.ROTATE_180), 
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]
    h_flip = cv2.flip(image, 1)
    v_flip = cv2.flip(image, 0)
    transpose = cv2.transpose(image)

    return rotated_images + [h_flip, v_flip, transpose]


def predict(model, image, device, transform):
    model.eval()
    with torch.no_grad():
        augmented = transform(image=image)
        image = augmented['image'].unsqueeze(0).to(device)

        output = model(image)
        prob = F.softmax(output, dim=1)
        prob = prob.cpu().numpy()

    return prob


def tta(model, data_path, csv_path, save_path, img_h, img_w, device):
    df = pd.read_csv(csv_path)
    transform = batch_transform(img_h, img_w)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        id = row['ID']
        img_path = f"{data_path}/{id}"
        image = cv2.imread(img_path)
        augmented_images = augment_image(image, 10, img_h, img_w)

        probs = []
        for aug_image in augmented_images:
            prob = predict(model, aug_image, device, transform)
            probs.append(prob)

        avg_prob = np.mean(probs, axis=0)
        target = np.argmax(avg_prob)
        df.at[idx, 'target'] = target

    df.to_csv(f"{save_path}/tta_submission.csv", index=False)

# def augment_image(image, n, img_h, img_w):
#     transforms = test_transform(img_h, img_w)
#     augmented_images = [transforms(image=image)['image'] for _ in range(n)]
#     return augmented_images


# def predict(model, image, device):
#     image = image.unsqueeze(0).to(device)
#     model.eval()
#     with torch.no_grad():
#         output = model(image)
#     prob = torch.nn.functional.softmax(output, dim=1)
#     return prob.cpu().numpy().flatten()


# def tta(model, data_path, csv_path, save_path, img_h, img_w, device, n_augmentations=10):
#     df = pd.read_csv(csv_path)

#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
#         id = row['ID']
#         img_path = f"{data_path}/{id}"
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         augmented_images = augment_image(image, n_augmentations, img_h, img_w)
        
#         probs = []
#         for aug_image in augmented_images:
#             prob = predict(model, aug_image, device)
#             probs.append(prob)

#         avg_prob = np.mean(probs, axis=0)
#         target = np.argmax(avg_prob)
#         df.at[idx, 'target'] = target

#     df.to_csv(f"{save_path}/tta_submission.csv", index=False)