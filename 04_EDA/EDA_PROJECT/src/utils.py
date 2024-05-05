import os
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

POS_FILTER = {
    'AM(CLR)'    : 'CAM, LAM, RAM',
    'M(CLR)'     : 'CM, LM, RM',
    'D(CLR)'     : 'CB, LB, RB',
    'AM(LR)'     : 'LAM, RAM',
    'AM(CL)'     : 'CAM, LAM',
    'AM(CR)'     : 'CAM, RAM',
    'AM(R)'      : 'CAM, RAM',
    'D(CL)'      : 'CB, LB',
    'D(LR)'      : 'LB, RB',
    'M(LR)'      : 'LM, RM',
    'D(CR)'      : 'CB, RB',
    'M(CR)'      : 'CM, RM',
    'M(CL)'      : 'CM, LM',
    'DMC'        : 'CDM',
    'AM(C)'      : 'CAM',
    'D(C)'       : 'CB',
    'M(R)'       : 'RM',
    'D(L)'       : 'LB',
    'D(R)'       : 'RB',
    'AM(L)'      : 'LAM',
    'M(C)'       : 'CM',
    'M(L)'       : 'LM',
    'FW'         : 'FW',
    'GK'         : 'GK',
    'Forward'    : 'FW',
    'Midfielder' : 'CM',
    'Goalkeeper' : 'GK',
    'Defender'   : 'CB'
}

# def map_positions(pos):
#     positions = [p.strip() for p in pos.split(',')]
#     mapped_positions = [POS_FILTER.get(p, p) for p in positions]
#     return ', '.join(mapped_positions)


def map_positions(pos):
    try:
        positions = [p.strip() for p in pos.split(',')]
        mapped_positions = [POS_FILTER.get(p, p) for p in positions]
        return ', '.join(mapped_positions)
    except Exception as e:
        # 에러가 발생하면, 로그에 기록하고 빈 문자열 반환
        print(f"An error occurred: {e}")
        return ''


def split_positions(df):
    position_split = df['Positions'].str.split(',', expand=True)
    position_split.columns = [f'Positions_{i+1}' for i in range(position_split.shape[1])]
    result_df = pd.concat([df, position_split], axis=1)
    
    return result_df


def team_name_mapper(player_df, team_df):
    player_teams_unique = sorted(player_df['Team_name'].unique())
    team_teams_unique = sorted(team_df['Team_name'].unique())

    team_name_dict = {}
    for pt, tt in zip(player_teams_unique, team_teams_unique):
        team_name_dict[pt] = tt

    return team_name_dict


def get_webdriver():
    browser = webdriver.Chrome(options=get_options())
    browser.implicitly_wait(3)

    return browser


def get_options():
    options = Options() 
    options.add_experimental_option("detach", True)

    return options


def close_ad(browser):
    close_button = WebDriverWait(browser, 1).until(EC.element_to_be_clickable((By.CLASS_NAME, "webpush-swal2-close")))
    close_button.click()


def make_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)