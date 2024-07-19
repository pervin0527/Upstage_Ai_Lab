import csv
import requests
import xml.etree.ElementTree as ET

# API 관련 변수 설정
API_KEY = '687a4575547065723536704a657150'  # 자신의 API 키로 변경하세요
SERVICE_NAME = 'tbLnOpendataRtmsV'
FILE_TYPE = 'json'
BASE_URL = 'http://data.seoul.go.kr/dataList/OA-21275/S/1/datasetView.do'  # 실제 API URL로 변경하세요

# CSV 파일 설정
OUTPUT_FILE = 'output.csv'

# 첫 번째 요청으로 총 데이터 건수 확인
def get_total_count():
    url = f"{BASE_URL}?KEY={API_KEY}&TYPE={FILE_TYPE}&SERVICE={SERVICE_NAME}&START_INDEX=1&END_INDEX=1"
    response = requests.get(url)
    
    if response.status_code == 200:
        try:
            tree = ET.ElementTree(ET.fromstring(response.content))
            root = tree.getroot()
            total_count = root.find('list_total_count').text
            return int(total_count)
        except Exception as e:
            print("Error parsing XML response:", e)
            return 0
    else:
        print("Error in API request:", response.status_code)
        return 0

# 데이터 가져오기 및 CSV 파일 저장
def fetch_and_save_data():
    total_count = get_total_count()
    if total_count == 0:
        print("No data to fetch")
        return

    start_index = 1
    end_index = 1000

    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'RCPT_YR', 'CGG_CD', 'CGG_NM', 'STDG_CD', 'STDG_NM', 'LOTNO_SE', 'LOTNO_SE_NM', 'MNO', 'SNO',
            'BLDG_NM', 'CTRT_DAY', 'THING_AMT', 'ARCH_AREA', 'LAND_AREA', 'FLR', 'RGHT_SE', 'RTRCN_DAY', 
            'ARCH_YR', 'BLDG_USG', 'DCLR_SE', 'OPBIZ_RESTAGNT_SGG_NM'
        ])  # 헤더 작성

        while start_index <= total_count:
            url = f"{BASE_URL}?KEY={API_KEY}&TYPE={FILE_TYPE}&SERVICE={SERVICE_NAME}&START_INDEX={start_index}&END_INDEX={end_index}"
            response = requests.get(url)
            
            if response.status_code == 200:
                try:
                    tree = ET.ElementTree(ET.fromstring(response.content))
                    root = tree.getroot()
                    rows = root.findall('row')
                    
                    for row in rows:
                        writer.writerow([
                            row.find('RCPT_YR').text if row.find('RCPT_YR') is not None else '',
                            row.find('CGG_CD').text if row.find('CGG_CD') is not None else '',
                            row.find('CGG_NM').text if row.find('CGG_NM') is not None else '',
                            row.find('STDG_CD').text if row.find('STDG_CD') is not None else '',
                            row.find('STDG_NM').text if row.find('STDG_NM') is not None else '',
                            row.find('LOTNO_SE').text if row.find('LOTNO_SE') is not None else '',
                            row.find('LOTNO_SE_NM').text if row.find('LOTNO_SE_NM') is not None else '',
                            row.find('MNO').text if row.find('MNO') is not None else '',
                            row.find('SNO').text if row.find('SNO') is not None else '',
                            row.find('BLDG_NM').text if row.find('BLDG_NM') is not None else '',
                            row.find('CTRT_DAY').text if row.find('CTRT_DAY') is not None else '',
                            row.find('THING_AMT').text if row.find('THING_AMT') is not None else '',
                            row.find('ARCH_AREA').text if row.find('ARCH_AREA') is not None else '',
                            row.find('LAND_AREA').text if row.find('LAND_AREA') is not None else '',
                            row.find('FLR').text if row.find('FLR') is not None else '',
                            row.find('RGHT_SE').text if row.find('RGHT_SE') is not None else '',
                            row.find('RTRCN_DAY').text if row.find('RTRCN_DAY') is not None else '',
                            row.find('ARCH_YR').text if row.find('ARCH_YR') is not None else '',
                            row.find('BLDG_USG').text if row.find('BLDG_USG') is not None else '',
                            row.find('DCLR_SE').text if row.find('DCLR_SE') is not None else '',
                            row.find('OPBIZ_RESTAGNT_SGG_NM').text if row.find('OPBIZ_RESTAGNT_SGG_NM') is not None else ''
                        ])
                except Exception as e:
                    print("Error parsing XML response:", e)
            else:
                print("Error in API request:", response.status_code)
            
            start_index += 1000
            end_index += 1000

if __name__ == "__main__":
    fetch_and_save_data()