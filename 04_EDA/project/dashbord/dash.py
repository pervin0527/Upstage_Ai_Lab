import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go

from util import *
from funcs import group_top_teams_by_position, group_lower_teams_by_position_per_league, find_similar_players, normalize

players_totals = pd.read_csv('./all_players_stats_total.csv')
teams_stats = pd.read_csv('./all_teams_stats.csv')

normalized_columns = players_totals.columns.drop(['Apps', 'player_name', 'team_name', 'position'])
players_totals[normalized_columns] = players_totals[normalized_columns].apply(normalize)

with open('./position_stats.json', 'r') as f:
    STATS_PER_POSITION = json.load(f)


#####################################################################################################
st.sidebar.title('Options')
TARGET = st.sidebar.selectbox('Positions', sorted(list(STATS_PER_POSITION.keys())), index=3)
N_MOST_SIMILAR_PLAYERS = st.sidebar.slider(label='Num Players', 
                                           min_value=1, 
                                           max_value=20, 
                                           value=10)

st.title('Finding cost-effective players in the transfer market')
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

top_market_value_player = top_players_stats_filtered.loc[top_players_stats_filtered['Market Value'].idxmax()]
top_market_value_player_name = top_market_value_player['player_name']
top_market_value_player_value = top_market_value_player['Market Value']
top_market_value_player_similarity = 1.0  ## 자기 자신과의 유사도는 1로 설정

# plot_data에 top_market_value_player 추가
new_row = pd.DataFrame({'player_name': [top_market_value_player_name],
                        'Market Value': [top_market_value_player_value],
                        'Similarity Score': [top_market_value_player_similarity]})

plot_data = pd.concat([plot_data, new_row], ignore_index=True)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=plot_data[plot_data['player_name'] != top_market_value_player_name]['Market Value'],
    y=plot_data[plot_data['player_name'] != top_market_value_player_name]['Similarity Score'],
    mode='markers+text',
    text=plot_data[plot_data['player_name'] != top_market_value_player_name]['player_name'],
    textposition='bottom center',
    textfont=dict(size=9),
    marker=dict(
        size=12,
        color=plot_data[plot_data['player_name'] != top_market_value_player_name]['Similarity Score'],
        colorscale='rainbow',
        showscale=True,
    ),
    hoverinfo='text',
    hovertext=plot_data[plot_data['player_name'] != top_market_value_player_name]['player_name'],
    name='Similar Players'
))

fig1.add_trace(go.Scatter(
    x=[top_market_value_player_value],
    y=[top_market_value_player_similarity],
    mode='markers+text',
    text=top_market_value_player_name,
    textposition='bottom center',
    marker=dict(
        size=15,
        color='red',
        symbol='star'
    ),
    hoverinfo='text',
    hovertext=top_market_value_player_name,
    name=top_market_value_player_name
))

fig1.update_layout(
    showlegend=False,
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
    title=dict(
        text="Similarity Score - Market Value",
        y=0.95,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=18)
    ),
    width=800,
    height=600
)

st.plotly_chart(fig1)
#####################################################################################################


## 시장 가치가 가장 높은 선수와 유사도 높은 선수들 선택
# top_market_value_player = similar_players_stats.loc[similar_players_stats['Market Value'].idxmax()]
merged_data = pd.merge(similar_players_stats, plot_data[['player_name', 'Similarity Score']], on='player_name', how='left')
sorted_data = merged_data.sort_values('Similarity Score', ascending=False)
top_n_similar_players = sorted_data.head(N_MOST_SIMILAR_PLAYERS)

## top_n_similar_players에서 top_market_value_player 제외
top_n_similar_players = top_n_similar_players[top_n_similar_players['player_name'] != top_market_value_player['player_name']]
selected_players_stats = pd.concat([top_market_value_player.to_frame().T, top_n_similar_players], ignore_index=True)

## selected_players_stats에서 중복되는 값 제외하고 가장 먼저 나오는 것만 남기기
selected_players_stats = selected_players_stats.drop_duplicates(subset='player_name', keep='first')
selected_player_names = st.sidebar.multiselect(
    'Select Players',
    options=selected_players_stats['player_name'].unique(),
    # default=selected_players_stats['player_name'].unique()[:2]
    default=selected_players_stats['player_name'].unique()[0]
)

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

selected_players_data = selected_players_stats[selected_players_stats['player_name'].isin(selected_player_names)]

fig2 = go.Figure()
for idx, player in enumerate(selected_players_data['player_name']):
    fig2.add_trace(go.Scatter(
        x=stat_columns,
        y=selected_players_data[stat_columns].iloc[idx].values.tolist(),
        mode='lines+markers',
        name=player,
        marker=dict(size=8),
        customdata=[player] * len(stat_columns),
        hovertemplate='%{customdata[0]}: %{y:.2f}<extra></extra>'
    ))

fig2.update_layout(
    xaxis=dict(
        tickangle=-40,
        tickfont=dict(size=10)
    ),
    yaxis=dict(
        range=[0, 1.1],
        tickfont=dict(size=10),
        fixedrange=True
    ),
    showlegend=True,
    legend=dict(
        orientation="h", 
        y=1.00,
        x=0.5,
        yanchor="bottom",
        xanchor="center", 
    ),
    title=dict(
        text="Most Expensive Player VS efficient Players",
        y=0.95,
        x=0.5, 
        xanchor='center',
        yanchor='top',
        font=dict(size=18)
    ),
    hovermode='closest',
    width=800,
    height=600,
    margin=dict(l=50, r=50, t=120, b=100)
)

st.plotly_chart(fig2)