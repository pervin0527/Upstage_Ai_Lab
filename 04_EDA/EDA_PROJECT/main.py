from src.transfermarket import market_value_crawling
from src.whoscored import league_table_crawling, team_stats_crawling, player_stats_crawling, player_detail_stats_crawling, clearing

if __name__ == "__main__":
    save_dir = '/Users/pervin0527/EDA_PROJECT/data'
    url = 'https://www.whoscored.com/'
    urls = [
        'https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/L1/plus/?saison_id=2022', ## 독일
        'https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1/plus/?saison_id=2022', ## 이탈리아
        'https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id=2022', ## 영국 
        'https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1/plus/?saison_id=2022', ## 스페인
        'https://www.transfermarkt.com/ligue-1/startseite/wettbewerb/FR1/plus/?saison_id=2022' ## 프랑스
    ]

    seasons = [
        '2022/2023'
    ]

    leagues = [
        'England-Premier-League', 
        'Spain-LaLiga', 
        'Italy-Serie-A', 
        'Germany-Bundesliga', 
        'France-Ligue-1',
    ]
    
    # league_table_crawling(url, leagues, seasons, save_dir)
    # team_stats_crawling(url, leagues, seasons, save_dir)
    # player_stats_crawling(url, leagues, seasons, save_dir)
    # player_detail_stats_crawling(url, leagues, seasons, save_dir)
    clearing(leagues, seasons, save_dir)

    market_value_crawling(urls, leagues, seasons, save_dir)