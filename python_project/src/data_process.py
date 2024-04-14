import pandas as pd
from collections import Counter

from src.keywords import *
from src.util import draw_main_graph, draw_sub_graph


def parse_and_count_associated_fields(field_counts, counter):
    for field_count in field_counts:
        try:
            for field, count in field_count.items():
                counter[field.strip()] += int(count)
        except ValueError:
            print(f"Skipping malformed entry: {field_count}")


def normalize_technologies(tech_details):
    words = tech_details.replace(',', ' ').split()
    normalized_words = [SYNONYMS.get(word, word) for word in words]
    return ' '.join(normalized_words)


def determine_field(words):
    fields = []
    for word in words:
        if word in IGNORE:
            continue
        elif word in CV:
            fields.append("컴퓨터 비전")
        elif word in NLP:
            fields.append("자연어 처리")
        elif word in DATA_ENGINEER:
            fields.append("데이터 엔지니어링")
        elif word in BACKEND:
            fields.append("백엔드 개발")
        elif word in FRONT:
            fields.append("프론트엔드 개발")
        elif word in MOBILE:
            fields.append("모바일 개발")
        elif word in BLOCK_CHAIN:
            fields.append("블록체인")
        elif word in FINANCE:
            fields.append("금융")
        elif word in SYSTEM:
            fields.append("시스템 개발")
        elif word in SECURITY:
            fields.append("보안")
        elif word in METAVERS:
            fields.append("메타버스")
    
    ## 모든 단어 검사 후 분야가 없으면 Unknown 추가
    if not fields:
        fields.append('Unknown')
    
    return ", ".join(sorted(set(fields)))


def count_technologies(row):
    words = row['기술 세부 사항'].replace(',', ' ').split()
    field_technologies = []

    field_lists = {
        "컴퓨터 비전": CV,
        "자연어 처리": NLP,
        "데이터 엔지니어링": DATA_ENGINEER,
        "백엔드 개발": BACKEND,
        "프론트엔드 개발": FRONT,
        "모바일 개발": MOBILE,
        "블록체인": BLOCK_CHAIN,
        "금융": FINANCE,
        "시스템 개발": SYSTEM,
        "보안": SECURITY,
        "메타버스": METAVERS,
        "Unknown": []
    }

    field_counts = {field: Counter([word for word in words if word in field_lists[field]]) for field in row['주요분야'].split(", ")}
    for field, counter in field_counts.items():
        for tech, count in counter.items():
            field_technologies.append({tech : count})
    
    return field_technologies


def preprocessing(*args):
    total_df = pd.concat(args)
    tech_details = total_df['기술 세부 사항']
    split_words = tech_details.apply(lambda x: x.replace(',', ' ').split())

    total_df['기술 세부 사항'] = total_df['기술 세부 사항'].apply(normalize_technologies)
    total_df['주요분야'] = split_words.apply(determine_field)
    total_df['연관 분야'] = total_df.apply(count_technologies, axis=1)

    field_counts = pd.Series([field for sublist in total_df['주요분야'].dropna().str.split(", ") for field in sublist]).value_counts()
    main_field_list = field_counts.index.to_list()
    main_field_counts = list(field_counts.values)
    
    main_field_list.pop(4)
    main_field_counts.pop(4)
    draw_main_graph(field_counts)

    related_field_list = [[] for _ in main_field_list]
    for i, keyword in enumerate(main_field_list):
        filtered_data = total_df[total_df['주요분야'].str.contains(keyword, na=False)]
        
        related_field_counter = Counter()
        filtered_data['연관 분야'].dropna().apply(lambda x: parse_and_count_associated_fields(x, related_field_counter))

        top_related_fields = related_field_counter.most_common(10)
        if top_related_fields:
            related_fields, related_counts = zip(*top_related_fields)
            related_field_list[i] = [{field: count} for field, count in zip(related_fields, related_counts)]
            draw_sub_graph(related_fields, related_counts, keyword)
        else:
            related_field_list[i] = []

    result = pd.DataFrame({'주요분야': main_field_list, '공고수': main_field_counts, '연관분야': related_field_list})
    result.to_csv('test.csv')