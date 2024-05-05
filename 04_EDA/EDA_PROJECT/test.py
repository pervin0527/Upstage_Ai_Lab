import pandas as pd

# 파일 경로
file_path = '/Users/pervin0527/EDA_PROJECT/data/Germany-Bundesliga/Germany-Bundesliga-2022_2023-players-stats.csv'

# 데이터 프레임으로 파일 읽기
df = pd.read_csv(file_path)

# 데이터 확인과 float 값 찾기
for index, row in df.iterrows():
    position = row['Positions']
    try:
        # 'Position' 값을 float로 변환 시도
        float_position = float(position)
        # 변환이 성공하면, float 값이 있는 것이므로 출력
        print(f"Row {index} contains a float in 'Position': {row}")
    except ValueError:
        # float 변환이 실패하면, 에러를 무시
        continue
