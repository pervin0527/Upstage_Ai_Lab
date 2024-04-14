import matplotlib.pyplot as plt

from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

def plot_top_keywords(top_vocabs, title, limit):
    """막대그래프 시각화"""
    keywords, frequencies = zip(*top_vocabs)
    plt.figure(figsize=(10, 20))
    plt.margins(x=0, y=0)

    plt.barh(keywords[:limit], frequencies[:limit], color='blue')
    plt.xlabel('빈도수')
    plt.ylabel('키워드')
    plt.title(title)
    plt.gca().invert_yaxis()
    # plt.show()

    plt.savefig(f"outputs/Top_{limit}.png")
    plt.close()


def plot_main_keywords(main_fields_dict):
    labels = list(main_fields_dict.keys())
    values = list(main_fields_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color='blue')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()//2, yval, round(yval), va='bottom')

    plt.title('분야별 데이터 수', fontsize=16)
    plt.xlabel('분야', fontsize=14)
    plt.ylabel('데이터 수', fontsize=14)

    plt.savefig("outputs/Final_Result.png")
    plt.close()


def plot_related_keywords(word, related_words_dict, idx):
    x_values = []
    y_values = []

    sorted_related_words_list = sorted(related_words_dict.items(), key=lambda x: x[1], reverse=True)
    for vocab, cnt in sorted_related_words_list[:10]:
        x_values.append(vocab)
        y_values.append(cnt)

    plt.figure(figsize=(12, 12))
    plt.bar(x_values, y_values)
    plt.xlabel('Vocabulary')
    plt.ylabel('Count')
    plt.title(f'"{word}"와 관련있는 최상위 10단어')

    plt.xticks(rotation=90)
    plt.savefig(f"outputs/Related_Result_{idx}.png")
    plt.close()