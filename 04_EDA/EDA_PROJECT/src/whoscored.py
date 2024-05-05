import time
import pandas as pd

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from src.utils import get_webdriver, close_ad, make_dir, map_positions, split_positions, team_name_mapper

TEAM_SUMMARY_COLUMNS = ['Team_name', 'Goals', 'Yellow_cards', 'Red_cards', 'Possessions(%)', 'Pass_Success(%)', 'A_Won', 'Rating']
TEAM_DEFENSIVE_COLUMNS = ['Team_name', 'Shot_allowed', 'Tackles_pg', 'Intercept_pg', 'Fouls_pg', 'Offsides_pg']
TEAM_OFFENSIVE_COLUMNS = ['Team_name', 'Shots_OT_pg', 'Dribbles_pg', 'Fouled_pg']
TEAM_XG_COLUMNS = ['Team_name', 'xG', 'Goals-OG', 'xGDiff', 'Shots', 'xG/shots']

DETAILED_SELECT_BOX = ['Tackles', 'Interception', 'Fouls', 'Cards', 'Offsides', 'Clearances', 'Blocks', 'Saves', 
                       'Shots', 'Goals', 'Dribbles', 'Possession loss', 'Aerial',
                       'Passes', 'Key passes', 'Assists']

# PLAYER_SUMMARY_COLUMNS = ['Player_name', 'Team_name', 'Age', 'Position', 'Apps', 'Mins', 'Goals', 'Assists', 'Yel', 'Red', 'Shots_pg', 'Pass_Success(%)', 'AerialsWon', 'MoM', 'Rating']


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
            team_table_df.to_csv(f"{curr_save_dir}/{league_name}-{seasons[0].replace('/', '_')}-teams-stats.csv", index=False)
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
            league_df = pd.read_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-teams-stats.csv')

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
                    summary_df = pd.DataFrame(table_data, columns=TEAM_SUMMARY_COLUMNS)


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
                    for curr_selected in DETAILED_SELECT_BOX:
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
                                column_headers[idx] = f'Total_{curr_selected}'
                            if column_header == 'SixYardBox':
                                column_headers[idx] = f'SixYardBox_{curr_selected}'
                            if column_header == 'PenaltyArea':
                                column_headers[idx] = f'PenaltyArea_{curr_selected}'
                            if column_header == 'OutOfBox':
                                column_headers[idx] = f'OutOfBox_{curr_selected}'
                

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
            league_df.to_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-teams-stats.csv', index=False)

    browser.close()


def player_stats_crawling(url, leagues, seasons, save_dir):
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
            browser.get(league_url)

            curr_save_dir = f"{save_dir}/{league_name}"
            make_dir(curr_save_dir)

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

            data_frames = []
            options = browser.find_element(By.ID, 'stage-top-player-stats-options').find_elements(By.TAG_NAME, 'li')
            for option in options[:-1]:
                option_name = option.text.lower()
                print(option_name)
                option.click()

                grid_container = browser.find_element(By.ID, f'statistics-mini-filter-{option_name}')
                grid_toolbar = grid_container.find_element(By.CLASS_NAME, 'grid-toolbar')
                grid_apps = grid_toolbar.find_element(By.ID, 'apps')
                grid_btns = grid_apps.find_elements(By.TAG_NAME, 'dd')
                target_btn = grid_btns[1].find_element(By.TAG_NAME, 'a')
                target_btn.click()
                time.sleep(3)

                paging_container = browser.find_element(By.ID, f'statistics-paging-{option_name}')
                current_page = int(paging_container.find_element(By.ID, "currentPage").get_attribute('value'))
                total_pages = int(paging_container.find_element(By.ID, "totalPages").get_attribute('value'))
                # total_pages = 2

                container = browser.find_element(By.ID, f'statistics-table-{option_name}')
                thead = container.find_element(By.TAG_NAME, 'thead')
                columns = thead.find_elements(By.TAG_NAME, 'th')
                columns = [column.text.strip() for column in columns[1:]]
                columns[0] = 'Player_name'

                if option_name == 'summary':
                    columns.insert(1, 'Team_name')
                    columns.insert(2, 'Age')
                    columns.insert(3, 'Positions')
                    columns[10] = 'Shoots_pg'
                    columns[11] = 'Pass_success(%)'
                else:
                    columns.pop(-1) ## Rating 제외
                    columns.pop(1)
                    columns.pop(1)

                if option_name == 'defensive':
                    columns[2] = 'Interceptions'
                    columns[6] = 'Dribbled_past_pg'
                elif option_name == 'offensive':
                    columns[3] = 'Shots_pg'
                    columns[4] = 'KeyPasses_pg'
                    columns[5] = 'Dribbles_pg'
                    columns[7] = 'Offsides_pg'
                    columns[8] = 'Dispossessed_pg'
                    columns[9] = 'Bad_Controls_pg'
                elif option_name == 'passing':
                    columns[2] = 'KeyPasses_pg'
                    columns[3] = 'Passes_pg'
                    columns[4] = 'Pass_success(%)'
                    columns[5] = 'Crosses_pg'
                    columns[6] = 'Long_pass_pg'
                    columns[7] = 'Through_pass_pg'
                elif option_name == 'xg':
                    columns[3] = 'Goals-xG'
                    columns[4] = 'xG_per_90m'
                    columns[5] = 'Total_shots'
                    columns[6] = 'xG_per_total_shots'

                print(columns)

                total_page_data = []
                while current_page <= total_pages:
                    time.sleep(5)
                    table = container.find_element(By.TAG_NAME, 'table')
                    table_body = table.find_element(By.TAG_NAME, 'tbody')
                    rows = table_body.find_elements(By.TAG_NAME, 'tr')
                    for row in rows:
                        row_idx = row.find_element(By.CLASS_NAME, 'table-ranking').text.strip()
                        spans = row.find_elements(By.TAG_NAME, 'span')
                        player_name = spans[0].text.strip()
                        row_data = [player_name]

                        if option_name == 'summary':
                            team_name = spans[2].text.strip()[:-1]
                            age = spans[4].text.strip()
                            positions = spans[5].text.split(',', 1)[1].strip()
                            row_data.extend([team_name, age, positions])

                        tds = row.find_elements(By.TAG_NAME, 'td')[2:]
                        tds = [td.text.strip() for td in tds]
                        tds = ['0' if x == '-' else x for x in tds]

                        if option_name != 'summary':
                            tds.pop(0)
                            tds.pop(0)
                            tds.pop(-1) ## Rating 제외

                        row_data.extend(tds)
                        total_page_data.append(row_data)
                        print(row_idx, ', '.join(row_data))

                    grid_toolbar = browser.find_element(By.ID, f'statistics-paging-{option.text.lower()}').find_element(By.CLASS_NAME, 'grid-toolbar')
                    rigth_box = grid_toolbar.find_element(By.CLASS_NAME, 'right')
                    next_button = rigth_box.find_element(By.ID, "next")
                    if next_button.is_enabled():
                        next_button.click()
                        current_page += 1  
                        
                print()
                df = pd.DataFrame(total_page_data, columns=columns)
                data_frames.append(df)

            ## 병합 전 중복 컬럼 제거
            final_df = data_frames[0]
            for df in data_frames[1:]:
                overlapping_columns = final_df.columns.intersection(df.columns)
                overlapping_columns = overlapping_columns.drop('Player_name')  ## Player_name 제외한 중복 컬럼들
                df.drop(columns=overlapping_columns, inplace=True)
                final_df = pd.merge(final_df, df, on='Player_name', how='outer')

            final_df.to_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-players-stats.csv', index=False)
    browser.close()


def player_detail_stats_crawling(url, leagues, seasons, save_dir):
    browser = get_webdriver()
    browser.get(url)
    wait_timer = WebDriverWait(browser, 5)

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
            league_df = pd.read_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-players-stats.csv')
            browser.get(league_url)
            time.sleep(2)

            select_box = Select(browser.find_element(By.ID,'seasons'))
            select_box.select_by_visible_text(seasons[0])

            if league_name == 'Italy-Serie-A':
                select_element = wait_timer.until(EC.element_to_be_clickable((By.ID, 'stages')))
                select_box = Select(select_element)
                select_box.select_by_visible_text("Serie A")  
            
            time.sleep(2)
            main_container = browser.find_element(By.CLASS_NAME, 'main-content-column')
            sub_nav = main_container.find_element(By.ID, 'sub-navigation')
            nav_items = sub_nav.find_elements(By.TAG_NAME, 'li')
            team_stat_btn = nav_items[3]
            team_stat_btn.click()

            option = browser.find_element(By.ID, 'stage-top-player-stats-options').find_elements(By.TAG_NAME, 'li')[-1]
            option.click()
            time.sleep(3)

            select_element = wait_timer.until(EC.element_to_be_clickable((By.ID, 'category')))
            select_box = Select(select_element)

            detailed_dfs = []
            for curr_selected in DETAILED_SELECT_BOX:
                print(curr_selected)
                select_box.select_by_visible_text(curr_selected)
                time.sleep(3)  # AJAX 로딩 대기

                paging_container = browser.find_element(By.ID, f'statistics-paging-detailed')
                current_page = int(paging_container.find_element(By.ID, "currentPage").get_attribute('value'))
                total_pages = int(paging_container.find_element(By.ID, "totalPages").get_attribute('value'))
                # total_pages = 2

                if current_page != 1:
                    grid_toolbar = browser.find_element(By.ID, f'statistics-paging-{option.text.lower()}').find_element(By.CLASS_NAME, 'grid-toolbar')
                    rigth_box = grid_toolbar.find_element(By.CLASS_NAME, 'right')
                    first_button = rigth_box.find_element(By.ID, "next")
                    if first_button.is_enabled():
                        first_button.click()

                container = browser.find_element(By.ID, 'statistics-table-detailed')
                table = container.find_element(By.ID, 'top-player-stats-summary-grid')

                thead = table.find_element(By.TAG_NAME, 'thead')
                column_headers = thead.find_elements(By.TAG_NAME, 'th')[1:-1]
                column_headers = [column.text.strip() for column in column_headers]
                column_headers[0] = 'Player_name'
                column_headers.pop(1)
                column_headers.pop(1)

                for idx, column_header in enumerate(column_headers):
                    if column_header == 'Total':
                        column_headers[idx] = f'Total_{curr_selected}'
                    if column_header == 'SixYardBox':
                        column_headers[idx] = f'SixYardBox_{curr_selected}'
                    if column_header == 'PenaltyArea':
                        column_headers[idx] = f'PenaltyArea_{curr_selected}'
                    if column_header == 'OutOfBox':
                        column_headers[idx] = f'OutOfBox_{curr_selected}'
                print(column_headers)

                table_data = []
                while current_page <= total_pages:
                    time.sleep(4)
                    table = container.find_element(By.ID, 'top-player-stats-summary-grid')
                    tbody = table.find_element(By.TAG_NAME, 'tbody')
                    rows = tbody.find_elements(By.TAG_NAME, 'tr')
                    for row in rows:
                        player_name = row.find_element(By.CLASS_NAME, 'iconize').text.strip()
                        tds = [cell.text.strip() for cell in row.find_elements(By.TAG_NAME, 'td')[4:-1]]
                        tds = ['0' if x == '-' else x for x in tds]

                        data_row = [player_name] + tds
                        print(', '.join(data_row))
                        table_data.append(data_row)

                    grid_toolbar = browser.find_element(By.ID, f'statistics-paging-{option.text.lower()}').find_element(By.CLASS_NAME, 'grid-toolbar')
                    rigth_box = grid_toolbar.find_element(By.CLASS_NAME, 'right')
                    next_button = rigth_box.find_element(By.ID, "next")
                    if next_button.is_enabled():
                        next_button.click()
                        current_page += 1  

                print()                
                category_df = pd.DataFrame(table_data, columns=column_headers)
                detailed_dfs.append(category_df)

            detailed_df = detailed_dfs[0]
            for df in detailed_dfs[1:]:
                detailed_df = detailed_df.merge(df, on='Player_name', how='outer')

            league_df = league_df.merge(detailed_df, on="Player_name", how='outer')
            league_df.to_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-players-stats-full.csv', index=False)
    browser.close()


def clearing(leagues, seasons, save_dir):
    for league_name in leagues:
        print(league_name)
        team_df = pd.read_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-teams-stats.csv')
        player_df = pd.read_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-players-stats.csv')

        ## 중복제거
        player_df.drop_duplicates(subset=['Player_name', 'Team_name'], keep='first', inplace=True)

        ## 포지션 세분화
        unique_positions_set = set()
        unique_positions = player_df['Positions'].unique()
        unique_positions_set.update(unique_positions)
        player_df['Positions'] = player_df['Positions'].apply(map_positions)

        position_words = set()
        for position in unique_positions_set:
            parts = position.split(',')
            for part in parts:
                cleaned_part = part.strip()
                position_words.add(cleaned_part)
        print(list(position_words))

        player_df = split_positions(player_df)

        ## teams_df와 players_df의 팀명 통일
        mapping_dict = team_name_mapper(player_df, team_df)
        player_df['Team_name'] = player_df['Team_name'].map(mapping_dict)

        ## 저장
        player_df.to_csv(f'{save_dir}/{league_name}/{league_name}-{seasons[0].replace("/", "_")}-players-stats-cleaned.csv', index=False)