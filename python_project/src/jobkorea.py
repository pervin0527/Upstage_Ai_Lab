import os
import time

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

class JobKoreaCrawler:
    def __init__(self, wait_sec=3, debug=True):
        self.url = 'https://www.jobkorea.co.kr/'
        self.wait_sec = wait_sec
        
        self.debug = debug
        self.options = Options()
        if debug:
            self.options.add_experimental_option("detach", True)


    def input_setup(self, browser):
        """검색조건 설정"""
        browser.find_element(By.ID, 'gnbGi').click()
        browser.find_element(By.XPATH, '//*[@id="devSearchForm"]/div[2]/div/div[1]/dl[1]/dt/p').click()
        browser.find_element(By.XPATH, '//*[@id="devSearchForm"]/div[2]/div/div[1]/dl[1]/dd[2]/div[2]/dl[1]/dd/div[1]/ul/li[6]/label/span').click()
        browser.find_element(By.XPATH, '//*[@id="duty_step2_10031_ly"]/li[14]/label/span').click()
        browser.find_element(By.ID, 'dev-btn-search').click()


    def get_data_from_page(self, browser, page_number):
        """현재 페이지에 있는 항목들에서 데이터 가져오기."""
        container = browser.find_element(By.CLASS_NAME, 'tplJobList').find_element(By.TAG_NAME, 'tbody')
        blocks = container.find_elements(By.TAG_NAME, 'tr')

        if len(blocks) < 10:
            return False

        current_page_data = []
        for i, block in tqdm(enumerate(blocks), desc=f"Page {page_number}", total=len(blocks), leave=True):
            try:
                company_block = block.find_element(By.CLASS_NAME, 'tplCo').find_element(By.TAG_NAME, 'a')
                company_name = company_block.text
                link = company_block.get_attribute('href')

                info_list = block.find_element(By.CLASS_NAME, 'titBx')
                title = info_list.find_element(By.CLASS_NAME, 'normalLog').text
                options = info_list.find_element(By.CLASS_NAME, 'etc').find_elements(By.CLASS_NAME, 'cell')
                options = [etc.text for etc in options[:-1]]
                options = ','.join(options)
                
                detail_fields = info_list.find_element(By.CLASS_NAME, 'dsc').text

                data = {'회사명' : company_name, '채용공고 제목' : title, '채용공고 세부 사항' : options, '기술 세부 사항' : detail_fields, '링크' : link}
                current_page_data.append(data)

            except NoSuchElementException as e:
                continue

        return current_page_data
    

    def crawling(self):
        """crawling을 수행하는 주요 함수."""
        browser = webdriver.Chrome(options=self.options)
        browser.get(self.url)
        self.input_setup(browser)

        idx = 1
        total_data = []
        while True:
            if self.debug and idx == 5:
                    break

            if idx > 1:
                url = f'https://www.jobkorea.co.kr/recruit/joblist?menucode=local&localorder=1#anchorGICnt_{idx}'
                try:
                    browser.get(url)
                except:
                    break
        
            time.sleep(self.wait_sec)
            page_data = self.get_data_from_page(browser, idx)

            if not page_data:
                break

            total_data.extend(page_data)
            idx += 1

        browser.close()
        return total_data