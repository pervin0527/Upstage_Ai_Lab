import re
import json
import unicodedata

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity


def normalize(column):
    max_val = column.max()
    min_val = column.min()
    if max_val == min_val:
        return column.apply(lambda x: 0)
    else:
        return (column - min_val) / (max_val - min_val)


def group_top_teams_by_position(all_players_stats, all_teams_stats):
    all_teams_stats['Pts'] = pd.to_numeric(all_teams_stats['Pts'])
    top_teams = all_teams_stats.nlargest(15, 'Pts')['team_name'].tolist()
    top_teams_df = all_players_stats[
        (all_players_stats['team_name'].isin(top_teams)) &
        (all_players_stats['Apps'].apply(lambda x: int(x.split('(')[0]) if pd.notna(x) else 0) >= 15)
    ].copy()
    top_teams_df['positions'] = top_teams_df['position'].apply(lambda x: [pos.strip() for pos in x.split(',') if pd.notna(x)])
    all_positions = set(pos for positions in top_teams_df['positions'] for pos in positions)
    position_groups = {pos: [] for pos in all_positions}
    for _, row in top_teams_df.iterrows():
        player_name = row['player_name']
        player_positions = row['positions']
        for pos in player_positions:
            position_groups[pos].append(player_name)
    return position_groups, top_teams


def group_lower_teams_by_position_per_league(all_players_stats, all_teams_stats):
    all_teams_stats['Pts'] = pd.to_numeric(all_teams_stats['Pts'])
    excluded_teams_per_league = all_teams_stats.groupby('League').apply(lambda x: x.nlargest(3, 'Pts')['team_name']).tolist()
    position_groups = {}
    for league in all_teams_stats['League'].unique():
        lower_teams = all_teams_stats[
            (all_teams_stats['League'] == league) &
            (~all_teams_stats['team_name'].isin(excluded_teams_per_league))
        ]['team_name'].tolist()
        lower_teams_df = all_players_stats[
            (all_players_stats['team_name'].isin(lower_teams)) &
            (all_players_stats['Apps'].apply(lambda x: int(x.split('(')[0]) if pd.notna(x) else 0) >= 15)
        ]
        for _, row in lower_teams_df.iterrows():
            positions = [pos.strip() for pos in row['position'].split(',') if pd.notna(row['position'])]
            player_name = row['player_name']
            for pos in positions:
                if pos in position_groups:
                    position_groups[pos].append(player_name)
                else:
                    position_groups[pos] = [player_name]
    return position_groups


# def find_similar_players(stats_per_position, top_players_stats, top_position_groups, bottom_players_stats, bottom_position_groups, position, n=10):
#     top_players_stats_filtered = top_players_stats[top_players_stats['player_name'].isin(top_position_groups[position])]
#     bottom_players_stats_filtered = bottom_players_stats[bottom_players_stats['player_name'].isin(bottom_position_groups[position])]
    
#     stat_columns = stats_per_position[position]
    
#     imputer = SimpleImputer(strategy='mean')
    
#     top_players_stats_imputed = imputer.fit_transform(top_players_stats_filtered[stat_columns])
#     top_players_median_stats = np.median(top_players_stats_imputed, axis=0).reshape(1, -1)
    
#     bottom_players_stats_imputed = imputer.transform(bottom_players_stats_filtered[stat_columns])
    
#     similarity_scores = cosine_similarity(bottom_players_stats_imputed, top_players_median_stats)
#     most_similar_indices = similarity_scores.flatten().argsort()[::-1][:n]
#     most_similar_players = bottom_players_stats_filtered.iloc[most_similar_indices]
    
#     return most_similar_players, similarity_scores.flatten()[most_similar_indices]

def find_similar_players(stats_per_position, top_players_stats, top_position_groups, bottom_players_stats, bottom_position_groups, position, n=10):
    top_players_stats_filtered = top_players_stats[top_players_stats['player_name'].isin(top_position_groups[position])]
    bottom_players_stats_filtered = bottom_players_stats[bottom_players_stats['player_name'].isin(bottom_position_groups[position])]
    
    top_player_with_max_value = top_players_stats_filtered.loc[top_players_stats_filtered['Market Value'].idxmax()]

    stat_columns = stats_per_position[position]
    imputer = SimpleImputer(strategy='mean')
    
    top_player_stats = top_player_with_max_value[stat_columns].values.reshape(1, -1)
    
    bottom_players_stats_imputed = imputer.fit_transform(bottom_players_stats_filtered[stat_columns])
    similarity_scores = cosine_similarity(bottom_players_stats_imputed, top_player_stats)
    
    most_similar_indices = similarity_scores.flatten().argsort()[::-1][:n]
    most_similar_players = bottom_players_stats_filtered.iloc[most_similar_indices]
    
    return most_similar_players, similarity_scores.flatten()[most_similar_indices]