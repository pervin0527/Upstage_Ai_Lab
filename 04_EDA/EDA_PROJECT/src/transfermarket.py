import requests
import pandas as pd

from bs4 import BeautifulSoup as bs

HEADERS = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}

def market_value_crawling(urls, leagues, seasons, save_dir):
    league_teams_url_list = []
    league_teams_name_list = []
    for url, league in zip(urls, leagues):
        req = requests.get(url, headers=HEADERS)
        soup = bs(req.text, 'html.parser')

        container = soup.find('div', id='yw1')
        table = container.find('table', class_='items')
        
        table_head = table.find('thead')
        headers = table_head.find('tr').find_all('th') ## 2 ~ 6
        # headers = headers[2:]
        headers = ['Club', 'Squad', 'Avg age', 'Foreigners', 'Avg market values', 'Total Market Values']

        table_body = table.find('tbody')
        table_rows = table_body.find_all('tr', class_=['odd', 'even'])

        team_ulrs = []
        team_names = []
        team_dataset = []
        for row in table_rows:
            team_name = row.find('td', class_='hauptlink').text.strip()
            team_names.append(team_name)

            team_url = row.find('td', class_='hauptlink').find('a').get('href')
            team_ulrs.append(f'https://www.transfermarkt.com{team_url}')

            squad = row.find_all('td', class_='zentriert')[1].text.strip()
            avg_age = row.find_all('td', class_='zentriert')[2].text.strip()
            foreigners = row.find_all('td', class_='zentriert')[3].text.strip()
            avg_market_values = row.find_all('td', class_='rechts')[0].text.strip()
            total_market_values = row.find_all('td', class_='rechts')[1].text.strip()
            team_dataset.append([team_name, squad, avg_age, foreigners, avg_market_values, total_market_values])

        team_df = pd.DataFrame(team_dataset, columns=headers)
        team_df.to_csv(f"{save_dir}/{league}/{league}-{seasons[0].replace('/', '_')}-teams-values.csv")

        league_teams_url_list.append(team_ulrs)
        league_teams_name_list.append(team_names)

    for idx, (team_names, team_urls) in enumerate(zip(league_teams_name_list, league_teams_url_list)):
        player_data = []
        for team_name, team_url in zip(team_names, team_urls):
            print(team_name, team_url)
            req = requests.get(team_url, headers=HEADERS)

            soup = bs(req.text, 'html.parser')
            container = soup.find('div', id='yw1')
            table = container.find('table', class_='items')
            table_body = table.find('tbody')

            table_rows = table_body.find_all('tr', class_=['odd', 'even'])

            for row in table_rows:  # 수정된 부분
                inline_table = row.find('td', class_='posrela')
                inline_table = inline_table.find('table', class_='inline-table')
                inner_table_rows = inline_table.find_all('tr')
                name = inner_table_rows[0].find_all('td')[1].text.strip()
                position = inner_table_rows[1].find('td').text.strip()

                dob_age_text = row.find_all('td', class_='zentriert')[1].text.strip()
                dob = ' '.join(dob_age_text.split()[:-1])
                age = dob_age_text.split()[-1].strip('()')

                nationality = row.find('img', class_='flaggenrahmen')['title']
                market_value = row.find('td', class_='rechts hauptlink').text.strip()

                player_data.append({
                    'Name':  name,
                    'Team' : team_name,
                    'Position': position,
                    'Date of Birth': dob,
                    'Age': age,
                    'Nationality': nationality,
                    'Market Value': market_value
                })

        league = leagues[idx]
        player_df = pd.DataFrame(player_data)
        player_df.to_csv(f"{save_dir}/{league}/{league}-{seasons[0].replace('/', '_')}-players-values.csv")