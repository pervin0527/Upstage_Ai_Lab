import time

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

class WantedCrawler:
    def __init__(self, keyword, wait_sec=3, debug=True):
        self.url = 'https://www.wanted.co.kr/'
        self.keyword = keyword
        self.wait_sec = wait_sec

        self.debug = debug
        self.options = Options()
        if debug:
            self.options.add_experimental_option("detach", True)


    def input_keyword(self, browser):
        """검색창에 keyword를 입력"""
        ## 네비게이션 바에서 [채용] 버튼 클릭.
        browser.find_element(By.CLASS_NAME, 'Menu_jobListLink__GYHAI').click()
        time.sleep(1.5)

        ## 직군에서 [개발] 선택
        browser.find_element(By.CLASS_NAME, 'JobGroup_JobGroup__H1m1m').click()
        browser.find_element(By.XPATH, '//*[@id="__next"]/div[3]/article/div/div[1]/section/ul/li[1]/a').click()

        ## [개발 분야] 선택
        browser.find_element(By.CLASS_NAME, 'JobCategory_JobCategory__btn__k3EFe').click()
        browser.find_element(By.XPATH, '//*[@id="__next"]/div[3]/article/div/div[2]/section/div[1]/div/button[9]').click()
        browser.find_element(By.CLASS_NAME, 'JobCategoryOverlay_JobCategoryOverlay__bottom__btn__GliIw').click()


    def scroll_down(self, browser):
        """스크롤을 내리는 함수."""

        idx = 0 ## For Debug mode
        last_height = browser.execute_script("return document.body.scrollHeight")
        while True:
            if self.debug and idx > 6:
                break
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            time.sleep(1.5)
            new_height = browser.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break

            last_height = new_height
            idx += 1 ## For Debug mode


    def crawling(self):
        """crawling을 수행하는 주요 함수."""
        browser = webdriver.Chrome(options=self.options)
        browser.get(self.url)
        self.input_keyword(browser)

        self.scroll_down(browser)

        container = browser.find_element(By.CLASS_NAME, 'List_List__FsLch')
        blocks = container.find_elements(By.CLASS_NAME, 'JobCard_JobCard__oZL4d')

        total_data = []
        for i, block in tqdm(enumerate(blocks), total=len(blocks), leave=False):
            try:
                card = block.find_element(By.TAG_NAME, 'a')
                company_name = card.get_attribute('data-company-name')
                title = card.get_attribute('data-position-name')
                link = card.get_attribute('href')

                data = {'회사명' : company_name, '채용공고 제목' : title, '링크' : link}
                total_data.append(data)

            except NoSuchElementException as e:
                print(f'\n\nBlock No : {i}, \n\t{e}')

        return total_data