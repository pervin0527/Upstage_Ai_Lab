import time
import pandas as pd

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from src.utils import get_webdriver, close_ad, make_dir

TEAM_SUMMARY_COLUMNS = ['Team_name', 'Goals', 'Yellow_cards', 'Red_cards', 'Possessions(%)', 'Pass_Success(%)', 'A_Won', 'Rating']
TEAM_DEFENSIVE_COLUMNS = ['Team_name', 'Shot_allowed', 'Tackles_pg', 'Intercept_pg', 'Fouls_pg', 'Offsides_pg']
TEAM_OFFENSIVE_COLUMNS = ['Team_name', 'Shots_OT_pg', 'Dribbles_pg', 'Fouled_pg']
TEAM_XG_COLUMNS = ['Team_name', 'xG', 'Goals-OG', 'xGDiff', 'Shots', 'xG/shots']

## excepted : 'Interception', 'Offsides'
TEAM_DETAILED_SELECT_BOX = ['Tackles', 'Fouls', 'Cards', 'Clearances', 'Blocks', 'Saves', 
                            'Shots', 'Goals', 'Dribbles', 'Possession loss', 'Aerial',
                            'Passes', 'Key passes', 'Assists']

PLAYER_SUMMARY_COLUMNS = ['Player_name', 'Team_name', 'Age', 'Position', 'Apps', 'Mins', 'Goals', 'Assists', 'Yel', 'Red', 'Shots_pg', 'Pass_Success(%)', 'AerialsWon', 'MoM', 'Rating']


def league_table_crawling(url, leagues, seasons, save_dir):
    browser = get_webdriver()
    browser.get(url)

    try:
        close_ad(browser)
    except:
        pass

    browser.find_element(By.ID, 'Top-Tournaments-btn').click()
    tournaments = browser.find_elements(By.CLASS_NAME, 'TournamentNavButton-module_clickableArea__ZFnBl')
    league_urls = [tournament.get_attribute('href') for tournament in tournaments]

    for league_url in league_urls:
        league_name = league_url.split('/')[-1]

        if league_name in leagues:
            print(league_name, league_url)
            browser.get(league_url)

            curr_save_dir = f"{save_dir}/{league_name}"
            make_dir(curr_save_dir)

            select_box = Select(browser.find_element(By.ID,'seasons'))
            select_box.select_by_visible_text(seasons[0])

            if league_name == 'Italy-Serie-A':
                select_element = WebDriverWait(browser, 10).until(EC.element_to_be_clickable((By.ID, 'stages')))
                select_box = Select(select_element)
                select_box.select_by_visible_text("Serie A")  

            container = browser.find_element(By.CLASS_NAME, 'tournament-standings-table')
            divs = container.find_elements(By.TAG_NAME, 'div')
            league_table = divs[1].find_element(By.TAG_NAME, 'tbody')
            rows = league_table.find_elements(By.TAG_NAME, 'tr')

            dataset = []
            columns=['Team_name', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
            for idx, row in enumerate(rows, start=1):
                team = row.find_element(By.CLASS_NAME, "team-link")
                team_name = team.text.strip()
                team_url = team.get_attribute('href')

                num_matches = int(row.find_element(By.CLASS_NAME, "p").text.strip())
                num_win = int(row.find_element(By.CLASS_NAME, 'w').text.strip())
                num_draw = int(row.find_element(By.CLASS_NAME, 'd').text.strip())
                num_lose = int(row.find_element(By.CLASS_NAME, 'l').text.strip())

                ## gf : 득점, ga : 실점, gd : 득실차, pts : 승점
                goal_for = int(row.find_element(By.CLASS_NAME, 'gf').text.strip())
                goal_against = int(row.find_element(By.CLASS_NAME, 'ga').text.strip())
                goal_difference = int(row.find_element(By.CLASS_NAME, 'gd').text.strip())
                points = int(row.find_element(By.CLASS_NAME, 'pts').text.strip())

                print(idx, team_name, num_matches, num_win, num_draw, num_lose, goal_for, goal_against, goal_difference, points)
                dataset.append([team_name, num_matches, num_win, num_draw, num_lose, goal_for, goal_against, goal_difference, points])
            
            team_table_df = pd.DataFrame(dataset, columns=columns)
            team_table_df.to_csv(f"{curr_save_dir}/{league_name}-{seasons[0].replace('/', '_')}-teams.csv", index=False)
            print()
    
    browser.close()


def team_stats_crawling(url, leagues, seasons, save_dir):
    browser = get_webdriver()
    browser.get(url)
    wait_timer = WebDriverWait(browser, 3)

    try:
        close_ad(browser)
    except:
        pass

    browser.find_element(By.ID, 'Top-Tournaments-btn').click()
    tournaments = browser.find_elements(By.CLASS_NAME, 'TournamentNavButton-module_clickableArea__ZFnBl')
    league_urls = [tournament.get_attribute('href') for tournament in tournaments]

    for league_url in league_urls:
        league_name = league_url.split('/')[-1]

        if league_name in leagues:
            print(league_name, league_url)
            league_df = pd.read_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-teams.csv')

            browser.get(league_url)

            select_box = Select(browser.find_element(By.ID,'seasons'))
            select_box.select_by_visible_text(seasons[0])

            if league_name == 'Italy-Serie-A':
                select_element = wait_timer.until(EC.element_to_be_clickable((By.ID, 'stages')))
                select_box = Select(select_element)
                select_box.select_by_visible_text("Serie A")  
            
            main_container = browser.find_element(By.CLASS_NAME, 'main-content-column')
            sub_nav = main_container.find_element(By.ID, 'sub-navigation')
            nav_items = sub_nav.find_elements(By.TAG_NAME, 'li')
            team_stat_btn = nav_items[2]
            team_stat_btn.click()

            options_container = wait_timer.until(EC.presence_of_element_located((By.ID, 'stage-team-stats-options')))
            options = options_container.find_elements(By.TAG_NAME, 'li')

            summary_df = pd.DataFrame()
            defensive_df = pd.DataFrame()
            offensive_df = pd.DataFrame()
            xg_df = pd.DataFrame()
            detailed_df = pd.DataFrame()
            for option in options:
                option_name = option.text.lower()
                option.click()

                container = browser.find_element(By.ID, f'stage-team-stats-{option_name}')
                print(f'stage-team-stats-{option_name}')
                table = container.find_element(By.CLASS_NAME, 'with-centered-columns')
                body = table.find_element(By.ID, 'top-team-stats-summary-content')
                rows = body.find_elements(By.TAG_NAME, 'tr')

                table_data = []
                if option_name == 'summary':
                    for idx, row in enumerate(rows, start=1):
                        team_name = row.find_element(By.CLASS_NAME, 'team-link').text.split(' ', 1)[1].strip()
                        goal = row.find_element(By.CLASS_NAME, 'goal').text
                        yellow_cards = row.find_element(By.CLASS_NAME, 'yellow-card-box').text
                        red_cards = row.find_element(By.CLASS_NAME, 'red-card-box').text
                        possession = row.find_element(By.CLASS_NAME, 'possession').text
                        pass_success = row.find_element(By.CLASS_NAME, 'passSuccess').text
                        aerial_won_per_game = row.find_element(By.CLASS_NAME, 'aerialWonPerGame').text
                        rating = row.find_element(By.CLASS_NAME, 'rating').text

                        print(idx, team_name, goal, yellow_cards, red_cards, possession, pass_success, aerial_won_per_game, rating)
                        table_data.append([team_name, goal, yellow_cards, red_cards, possession, pass_success, aerial_won_per_game, rating])
                    TEAM_summary_df = pd.DataFrame(table_data, columns=TEAM_SUMMARY_COLUMNS)


                if option_name == 'defensive':
                    for idx, row in enumerate(rows, start=1):
                        team_name = row.find_element(By.CLASS_NAME, 'team-link').text.split(' ', 1)[1].strip()
                        # shots_pg = row.find_element(By.CLASS_NAME, 'shotsConcededPerGame').text
                        shots_allowed = row.find_element(By.CLASS_NAME, 'shotsConcededPerGame').text
                        tackles_pg = row.find_element(By.CLASS_NAME, 'tacklePerGame').text
                        interceptions_pg = row.find_element(By.CLASS_NAME, 'interceptionPerGame').text
                        fouls_pg = row.find_element(By.CLASS_NAME, 'foulsPerGame').text
                        offsides_pg = row.find_element(By.CLASS_NAME, 'offsideGivenPerGame').text

                        print(idx, team_name, shots_allowed, tackles_pg, interceptions_pg, fouls_pg, offsides_pg)
                        table_data.append([team_name, shots_allowed, tackles_pg, interceptions_pg, fouls_pg, offsides_pg])
                    defensive_df = pd.DataFrame(table_data, columns=TEAM_DEFENSIVE_COLUMNS)


                if option_name == 'offensive':
                    columns = ['team_name', 'Shots OT pg', 'Dribbles pg', 'Fouled pg']
                    for idx, row in enumerate(rows, start=1):
                        team_name = row.find_element(By.CLASS_NAME, 'team-link').text.split(' ', 1)[1].strip()
                        shot_on_target_per_game = row.find_element(By.CLASS_NAME, 'shotOnTargetPerGame').text
                        dribble_won_per_game = row.find_element(By.CLASS_NAME, 'dribbleWonPerGame').text
                        foul_given_per_game = row.find_element(By.CLASS_NAME, 'foulGivenPerGame').text

                        print(idx, team_name, shot_on_target_per_game, dribble_won_per_game, foul_given_per_game)
                        table_data.append([team_name, shot_on_target_per_game, dribble_won_per_game, foul_given_per_game])
                    offensive_df = pd.DataFrame(table_data, columns=TEAM_OFFENSIVE_COLUMNS)
                

                if option_name == 'xg':
                    for idx, row in enumerate(rows, start=1):
                        team_name = row.find_element(By.CLASS_NAME, 'team-link').text.split(' ', 1)[1].strip()
                        xg = row.find_element(By.CLASS_NAME, 'xG').text
                        goals_diff = row.find_element(By.CLASS_NAME, 'goalExcOwn').text
                        xgdiff = row.find_element(By.CLASS_NAME, 'xGDiff').text
                        total_shots = row.find_element(By.CLASS_NAME, 'totalShots').text
                        xg_per_shot = row.find_element(By.CLASS_NAME, 'xGPerShot').text

                        print(idx, team_name, xg, goals_diff, xgdiff, total_shots, xg_per_shot)
                        table_data.append([team_name, xg, goals_diff, xgdiff, total_shots, xg_per_shot])
                    xg_df = pd.DataFrame(table_data, columns=TEAM_XG_COLUMNS)


                if option_name == 'detailed':
                    select_element = wait_timer.until(EC.element_to_be_clickable((By.ID, 'category')))
                    select_box = Select(select_element)

                    detailed_dfs = []
                    for curr_selected in TEAM_DETAILED_SELECT_BOX:
                        print(curr_selected)
                        select_box.select_by_visible_text(curr_selected)
                        time.sleep(1.5)  # AJAX 로딩 대기

                        container = browser.find_element(By.ID, 'statistics-team-table-detailed')
                        table = container.find_element(By.ID, 'top-team-stats-summary-grid')

                        theads = table.find_element(By.TAG_NAME, 'thead').find_elements(By.TAG_NAME, 'th')

                        end_line = -1
                        if curr_selected == 'Fouls':
                            end_line = -2

                        column_headers = [thead.text for thead in theads[:end_line]]
                        column_headers[0] = 'Team_name'

                        for idx, column_header in enumerate(column_headers):
                            if column_header == 'Total':
                                column_headers[idx] = f'{curr_selected}_Total'
                            if column_header == 'SixYardBox':
                                column_headers[idx] = f'{curr_selected}_SixYardBox'
                            if column_header == 'PenaltyArea':
                                column_headers[idx] = f'{curr_selected}_PenaltyArea'
                            if column_header == 'OutOfBox':
                                column_headers[idx] = f'{curr_selected}_OutOfBox'
                

                        tbody = table.find_element(By.TAG_NAME, 'tbody')
                        rows = tbody.find_elements(By.TAG_NAME, 'tr')
                        
                        table_data = []
                        for row in rows:
                            team_name = row.find_element(By.CLASS_NAME, 'team-link').text.split(' ', 1)[1].strip()
                            data_row = [team_name] + [cell.text for cell in row.find_elements(By.TAG_NAME, 'td')[1:end_line]]
                            table_data.append(data_row)

                        print(column_headers)
                        for data in table_data:
                            print(data)
                        print()
                        
                        category_df = pd.DataFrame(table_data, columns=column_headers)
                        detailed_dfs.append(category_df)

                    detailed_df = detailed_dfs[0]
                    for df in detailed_dfs[1:]:
                        detailed_df = detailed_df.merge(df, on='Team_name', how='outer')

                print()

            league_df = league_df.merge(summary_df, on='Team_name', how='outer')
            league_df = league_df.merge(defensive_df, on='Team_name', how='outer')
            league_df = league_df.merge(offensive_df, on='Team_name', how='outer')
            league_df = league_df.merge(xg_df, on='Team_name', how='outer')
            league_df = league_df.merge(detailed_df, on='Team_name', how='outer')
            league_df.to_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-teams.csv', index=False)

    browser.close()


def player_stats_crawling(url, leagues, seasons, save_dir):
    browser = get_webdriver()
    browser.get(url)
    wait_timer = WebDriverWait(browser, 1.5)

    try:
        close_ad(browser)
    except:
        pass

    browser.find_element(By.ID, 'Top-Tournaments-btn').click()
    tournaments = browser.find_elements(By.CLASS_NAME, 'TournamentNavButton-module_clickableArea__ZFnBl')
    league_urls = [tournament.get_attribute('href') for tournament in tournaments]

    for league_url in league_urls:
        league_name = league_url.split('/')[-1]

        if league_name in leagues:
            print(league_name, league_url)
            league_df = pd.read_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-teams.csv')

            browser.get(league_url)

            select_box = Select(browser.find_element(By.ID,'seasons'))
            select_box.select_by_visible_text(seasons[0])

            if league_name == 'Italy-Serie-A':
                select_element = wait_timer.until(EC.element_to_be_clickable((By.ID, 'stages')))
                select_box = Select(select_element)
                select_box.select_by_visible_text("Serie A")  
            
            main_container = browser.find_element(By.CLASS_NAME, 'main-content-column')
            sub_nav = main_container.find_element(By.ID, 'sub-navigation')
            nav_items = sub_nav.find_elements(By.TAG_NAME, 'li')
            team_stat_btn = nav_items[3]
            team_stat_btn.click()

            options = browser.find_element(By.ID, 'stage-top-player-stats-options').find_elements(By.TAG_NAME, 'li')
            for option in options:
                option_name = option.text.lower()
                option.click()
                # time.sleep(1.5)

                grid_container = browser.find_element(By.ID, f'statistics-mini-filter-{option_name}')
                grid_toolbar = grid_container.find_element(By.CLASS_NAME, 'grid-toolbar')
                grid_apps = grid_toolbar.find_element(By.ID, 'apps')
                grid_btns = grid_apps.find_elements(By.TAG_NAME, 'dd')
                target_btn = grid_btns[1].find_element(By.TAG_NAME, 'a')
                target_btn.click()
                time.sleep(2)

                paging_container = browser.find_element(By.ID, f'statistics-paging-{option_name}')
                current_page = int(paging_container.find_element(By.ID, "currentPage").get_attribute('value'))
                total_pages = int(paging_container.find_element(By.ID, "totalPages").get_attribute('value')) + 1
                # print(current_page, total_pages)

                total_page_data = []
                container = browser.find_element(By.ID, f'statistics-table-{option_name}')
                while current_page < total_pages:
                    time.sleep(1.5)
                    table = container.find_element(By.TAG_NAME, 'table')
                    table_body = table.find_element(By.TAG_NAME, 'tbody')
                    rows = table_body.find_elements(By.TAG_NAME, 'tr')

                    if option_name == 'summary':
                        for row in rows:
                            row_idx = row.find_element(By.CLASS_NAME, 'table-ranking').text.strip()
                            spans = row.find_elements(By.TAG_NAME, 'span')
                            player_name = spans[0].text.strip()
                            team_name = spans[2].text.strip()[:-1]
                            age = spans[4].text.strip()
                            positions = spans[5].text.split(',', 1)[1].strip()
                            row_data = [player_name, team_name, age, positions]

                            tds = row.find_elements(By.TAG_NAME, 'td')[2:]
                            tds = [td.text.strip() for td in tds]
                            tds = ['0' if x == '-' else x for x in tds]
                            row_data.extend(tds)

                            total_page_data.append(row_data)
                            print(row_idx, ', '.join(row_data))

                    grid_toolbar = browser.find_element(By.ID, f'statistics-paging-{option.text.lower()}').find_element(By.CLASS_NAME, 'grid-toolbar')
                    rigth_box = grid_toolbar.find_element(By.CLASS_NAME, 'right')
                    next_button = rigth_box.find_element(By.ID, "next")
                    if next_button.is_enabled():
                        next_button.click()
                        current_page += 1              

                if option_name == "summary":
                    df = pd.DataFrame(total_page_data, columns=PLAYER_SUMMARY_COLUMNS)
                    df.to_csv('/Users/pervin0527/EDA_PROJECT/test.csv', index=False)
                
                break
            break
        break
    browser.close()