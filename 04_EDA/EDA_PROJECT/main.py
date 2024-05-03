from src.whoscored import league_table_crawling, team_stats_crawling, player_stats_crawling

if __name__ == "__main__":
    url = 'https://www.whoscored.com/'
    save_dir = '/Users/pervin0527/EDA_PROJECT/data'

    seasons = [
        '2022/2023'
    ]

    leagues = [
        'England-Premier-League', 
        'Spain-LaLiga', 
        'Italy-Serie-A', 
        'Germany-Bundesliga', 
        'France-Ligue-1'
    ]
    
    # league_table_crawling(url, leagues, seasons, save_dir)
    # team_stats_crawling(url, leagues, seasons, save_dir)
    player_stats_crawling(url, leagues, seasons, save_dir)