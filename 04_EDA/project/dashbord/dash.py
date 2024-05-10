import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go

from util import *
from adjustText import adjust_text
from funcs import group_top_teams_by_position, group_lower_teams_by_position_per_league, find_similar_players, normalize

players_totals = pd.read_csv('./all_players_stats_total.csv')
teams_stats = pd.read_csv('./all_teams_stats.csv')
with open('./position_stats.json', 'r') as f:
    STATS_PER_POSITION = json.load(f)

st.title('이적시장에서 가성비 선수 찾기')

# 사이드바 메뉴 생성
st.sidebar.title('옵션')
TARGET = st.sidebar.selectbox('포지션', sorted(list(STATS_PER_POSITION.keys())), index=3)
N_MOST_SIMILAR_PLAYERS = st.sidebar.slider(label='선수 숫자', 
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

fig1 = go.Figure(data=go.Scatter(
    x=plot_data['Market Value'],
    y=plot_data['Similarity Score'],
    mode='markers+text',
    text=plot_data['player_name'],
    textposition='top center',
    marker=dict(
        size=10,
        color=plot_data['Similarity Score'],
        colorscale='Viridis',
        showscale=True
    ),
    hoverinfo='text',
    hovertext=plot_data['player_name']
))

fig1.update_layout(
    xaxis=dict(
        title='Market Value',
        tickformat=',.0f€',
        gridcolor='lightgrey',
        gridwidth=0.5
    ),
    yaxis=dict(
        title='Similarity Score',
        gridcolor='lightgrey',
        gridwidth=0.5
    ),
    title=f'포지션 : {TARGET} (x축 : 시장가치, y축 : 유사도)',
    width=800,
    height=600
)

st.plotly_chart(fig1)


## 시장 가치가 가장 높은 선수와 유사도 높은 선수들 선택
top_market_value_player = similar_players_stats.loc[similar_players_stats['Market Value'].idxmax()]
merged_data = pd.merge(similar_players_stats, plot_data[['player_name', 'Similarity Score']], on='player_name', how='left')
sorted_data = merged_data.sort_values('Similarity Score', ascending=False)
top_n_similar_players = sorted_data.head(5)

## top_n_similar_players에서 top_market_value_player 제외
top_n_similar_players = top_n_similar_players[top_n_similar_players['player_name'] != top_market_value_player['player_name']]

## selected_players_stats에서 중복되는 값 제외하고 가장 먼저 나오는 것만 남기기
selected_players_stats = pd.concat([top_market_value_player.to_frame().T, top_n_similar_players], ignore_index=True)
selected_players_stats = selected_players_stats.drop_duplicates(subset='player_name', keep='first')

## 선택된 선수들의 스탯 비교 레이더 차트 그리기
if TARGET == 'Goalkeeper':
    stat_columns = GK_STATS
elif TARGET == 'Centre-Back':
    stat_columns = CB_STATS
elif TARGET == 'Right-Back':
    stat_columns = RB_STATS
elif TARGET == 'Left-Back':
    stat_columns = LB_STATS
elif TARGET == 'Central Midfield':
    stat_columns = CM_STATS
elif TARGET == 'Attacking Midfield':
    stat_columns = AM_STATS
elif TARGET == 'Defensive Midfield':
    stat_columns = DM_STATS
elif TARGET == 'Left Midfield':
    stat_columns = LM_STATS
elif TARGET == 'Right Midfield':
    stat_columns = RM_STATS
elif TARGET == 'Right Winger':
    stat_columns = RW_STATS
elif TARGET == 'Left Winger':
    stat_columns = LW_STATS
elif TARGET == 'Centre-Forward':
    stat_columns = ST_STATS
selected_players_stats_normalized = selected_players_stats[stat_columns].apply(normalize)

fig2 = go.Figure()

for idx, player in enumerate(selected_players_stats['player_name']):
    fig2.add_trace(go.Scatterpolar(
        r=selected_players_stats_normalized.iloc[idx].values.tolist(),
        theta=stat_columns,
        name=player,
        customdata=[[player]] * len(stat_columns),
        hovertemplate='%{customdata[0]}<extra></extra>',
        fill='none'
    ))

fig2.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        ),
        angularaxis=dict(
            tickfont=dict(size=12)
        )
    ),
    showlegend=True,
    legend=dict(
        orientation="h",
        y=1.05,
        x=0.5,
        yanchor="bottom",
        xanchor="center",
    ),
    title=dict(
        text="가장 비싼 선수 VS 가성비 선수",
        y=0.99,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=18)
    ),
    hovermode='closest',
    hoverlabel=dict(
        namelength=-1
    ),
    width=750,
    height=750,
    margin=dict(l=50, r=50, t=100, b=100)
)

for trace in fig2.data:
    trace.update(line=dict(dash=None, width=2), marker=dict(size=5))

st.plotly_chart(fig2)