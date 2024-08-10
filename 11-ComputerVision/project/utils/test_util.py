import os
import cv2
import timm
import math
import torch
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

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
        augmented_images = augment_image(image)

        probs = []
        for aug_image in augmented_images:
            prob = predict(model, aug_image, device, transform)
            probs.append(prob)

        avg_prob = np.mean(probs, axis=0)
        target = np.argmax(avg_prob)
        df.at[idx, 'target'] = target

    df.to_csv(f"{save_path}/tta_submission.csv", index=False)


def read_coordinates_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    if not lines:
        return None  # 파일이 비어 있을 경우 None 반환
    return lines


def calculate_rotation_angle(coord):
    # 좌표가 다음과 같은 순서로 제공된다고 가정: [x1, y1, x2, y2, x3, y3, x4, y4]
    # x1, y1: 좌측 상단 / x2, y2: 우측 상단 / x3, y3: 우측 하단 / x4, y4: 좌측 하단
    
    dx = coord[2] - coord[0]
    dy = coord[3] - coord[1]
    
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def crop_text_regions(image_path, coordinates):
    image = cv2.imread(image_path)
    
    cropped_images = []
    for i, coord in enumerate(coordinates):
        coord = [int(x) for x in coord.split(',')]
        
        # 회전 각도 계산
        angle = calculate_rotation_angle(coord)
        
        # 이미지 회전 보정
        rotated_image = rotate_image(image, angle)
        
        left = min(coord[0], coord[6])
        top = min(coord[1], coord[3])
        right = max(coord[2], coord[4])
        bottom = max(coord[5], coord[7])
        
        cropped = rotated_image[top:bottom, left:right]
        cropped_images.append(cropped)
    
    return cropped_images


def test_with_ocr(model, data_path, csv_path, save_path, img_h, img_w, device):
    problem_list = [1, 3, 4, 6, 7, 10, 11, 12, 13, 14]
    processor = TrOCRProcessor.from_pretrained("team-lucid/trocr-small-korean")
    ocr_model = VisionEncoderDecoderModel.from_pretrained("team-lucid/trocr-small-korean").to(device)

    df = pd.read_csv(csv_path)
    transform = batch_transform(img_h, img_w)

    keywords = {
        1: ["임신"],
        3: ["입퇴원", "입원", "퇴원"],
        4: ["진단"],
        6: ["의료비 영수증", "의료비"],
        7: ["외래", "진료 확인서", "통원확인서", "치료확인서"],
        10: ["진료비"],
        11: ["조제약", "약국"],
        12: ["처방전", "처방"],
        13: ["이력서", "지원서"],
        14: ["소견서"]
    }

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        id = row['ID']
        img_path = f"{data_path}/{id}"
        image = cv2.imread(img_path)
        augmented_images = augment_image(image)

        probs = []
        for aug_image in augmented_images:
            prob = predict(model, aug_image, device, transform)
            probs.append(prob)

        avg_prob = np.mean(probs, axis=0)
        target = np.argmax(avg_prob)
        df.at[idx, 'target'] = target

        file_name = id.split('.')[0]
        if target in problem_list:
            coord_path = f"/home/pervinco/upstage-cv-classification-cv7/dataset/test_with_bbox/res_{file_name}.txt"
            coordinates = read_coordinates_from_file(coord_path)
            
            if coordinates is None:
                continue
            
            crop_imgs = crop_text_regions(img_path, coordinates)
            
            if len(crop_imgs) > 0:
                extracted_texts = []
                for i, crop_img in enumerate(crop_imgs):
                    pixel_values = processor(crop_img, return_tensors="pt").pixel_values.to(device)
                    generated_ids = ocr_model.generate(pixel_values)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    extracted_texts.append(generated_text)

                # 키워드가 추출된 텍스트에 포함되어 있는지 확인하고 target 업데이트
                for text in extracted_texts:
                    for key, value_list in keywords.items():
                        if any(keyword in text for keyword in value_list):
                            df.at[idx, 'target'] = key
                            break  # 키워드를 찾았으면 더 이상 반복할 필요 없음

    df.to_csv(f"{save_path}/ocr_submission.csv", index=False)
