import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


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