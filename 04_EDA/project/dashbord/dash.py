import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from adjustText import adjust_text
from funcs import group_top_teams_by_position, group_lower_teams_by_position_per_league, find_similar_players, normalize

players_totals = pd.read_csv('./all_players_stats_total.csv')
teams_stats = pd.read_csv('./all_teams_stats.csv')
with open('./position_stats.json', 'r') as f:
    STATS_PER_POSITION = json.load(f)

st.title('Football Player Similarity Analysis')
TARGET = st.selectbox('Select a position', sorted(list(STATS_PER_POSITION.keys())), index=3)
N_MOST_SIMILAR_PLAYERS = st.slider(label='Number of most similar players', 
                                   min_value=50, 
                                   max_value=100, 
                                   value=50, 
                                   step=5)


## 상위권 선수들, 하위권 선수들을 포지션별로 그룹화.
top_position_groups, top_teams = group_top_teams_by_position(players_totals, teams_stats)
bottom_position_groups = group_lower_teams_by_position_per_league(players_totals, teams_stats)
top_players_stats = players_totals[players_totals['team_name'].isin(top_teams)]
bottom_players_stats = players_totals[~players_totals['team_name'].isin(top_teams)]

## 유사 선수 찾기
most_similar_players, similarity_scores = find_similar_players(STATS_PER_POSITION, top_players_stats, top_position_groups, bottom_players_stats, bottom_position_groups, TARGET, N_MOST_SIMILAR_PLAYERS)
top_players_stats_filtered = top_players_stats[top_players_stats['player_name'].isin(top_position_groups[TARGET])]
similar_players_stats = bottom_players_stats[bottom_players_stats['player_name'].isin(most_similar_players['player_name'])]

## 시장 가치, 유사도 그래프 그리기
plot_data = most_similar_players[['player_name', 'Market Value']].copy()
plot_data['Similarity Score'] = similarity_scores
plot_data.dropna(subset=['Market Value', 'Similarity Score'], inplace=True)

unique_players = plot_data['player_name'].unique()
color_map = plt.cm.get_cmap('rainbow', len(unique_players))
player_colors = {player: color_map(i) for i, player in enumerate(unique_players)}

fig1, ax1 = plt.subplots(figsize=(15, 15))
scatter = plt.scatter(x=plot_data['Market Value'], y=plot_data['Similarity Score'], s=100, c=[player_colors[name] for name in plot_data['player_name']], alpha=0.6)

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in player_colors.values()]
ax1.legend(handles, player_colors.keys(), title='Players', loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

texts = []
for i, row in plot_data.iterrows():
    texts.append(ax1.text(row['Market Value'], row['Similarity Score'], row['player_name']))
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}€'.format(x)))
ax1.set_xlabel('Market Value')
ax1.set_ylabel('Similarity Score')
ax1.set_title(f'Similarity Score vs. Market Value for Top {N_MOST_SIMILAR_PLAYERS} Similar {TARGET} Players')
ax1.grid(True)

st.pyplot(fig1)

## 시장 가치가 가장 높은 선수와 유사도 높은 선수들 선택
top_market_value_player = similar_players_stats.loc[similar_players_stats['Market Value'].idxmax()]
merged_data = pd.merge(similar_players_stats, plot_data[['player_name', 'Similarity Score']], on='player_name', how='left')
sorted_data = merged_data.sort_values('Similarity Score', ascending=False)
top_n_similar_players = sorted_data.head(10)

# top_n_similar_players에서 top_market_value_player 제외
top_n_similar_players = top_n_similar_players[top_n_similar_players['player_name'] != top_market_value_player['player_name']]

# selected_players_stats에서 중복되는 값 제외하고 가장 먼저 나오는 것만 남기기
selected_players_stats = pd.concat([top_market_value_player.to_frame().T, top_n_similar_players], ignore_index=True)
selected_players_stats = selected_players_stats.drop_duplicates(subset='player_name', keep='first')

## 선택된 선수들의 스탯 비교 레이더 차트 그리기
stat_columns = STATS_PER_POSITION[TARGET] + ['age', 'Mins']
selected_players_stats_normalized = selected_players_stats[stat_columns].apply(normalize)
angles = np.linspace(0, 2 * np.pi, len(stat_columns), endpoint=False).tolist()
angles += angles[:1]

fig2, ax2 = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_players_stats)))
lines = []
for i, player in enumerate(selected_players_stats['player_name']):
    player_stats = selected_players_stats_normalized.iloc[i].values.flatten().tolist()
    player_stats += player_stats[:1]
    line, = ax2.plot(angles, player_stats, linewidth=1, linestyle='--', marker='o', color=colors[i], label=player, picker=5)
    lines.append(line)

ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)
ax2.set_rlabel_position(70)
ax2.set_thetagrids(np.degrees(angles[:-1]), labels=stat_columns, fontsize=7)
ax2.set_title("Player Stats Comparison", fontsize=8)
ax2.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=12)
plt.tight_layout()
st.pyplot(fig2)