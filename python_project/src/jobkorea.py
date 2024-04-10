import time

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

class JobKoreaCrawler:
    def __init__(self, keyword, wait_sec=3, debug=True):
        self.url = 'https://www.jobkorea.co.kr/'
        self.keyword = keyword
        self.wait_sec = wait_sec
        
        self.debug = debug
        self.options = Options()
        if debug:
            self.options.add_experimental_option("detach", True)


    def input_keyword(self, browser):
        """검색창에 keyword를 입력."""
        input_box = browser.find_element(By.CLASS_NAME, 'smKey').find_element(By.TAG_NAME, 'input')
        input_box.send_keys(self.keyword + Keys.RETURN)


    def get_data_from_page(self, browser, page_number):
        """현재 페이지에 있는 항목들에서 데이터 가져오기."""
        container = browser.find_element(By.CLASS_NAME, 'list-default')
        blocks = container.find_elements(By.CLASS_NAME, 'list-post')

        current_page_data = []
        for i, block in tqdm(enumerate(blocks), desc=f"Page {page_number}", total=len(blocks), leave=False):
            try:
                company_name = block.find_element(By.CLASS_NAME, 'post-list-corp').find_element(By.TAG_NAME, 'a').text
                list_info = block.find_element(By.CLASS_NAME, 'post-list-info').find_element(By.TAG_NAME, 'a')
                title = list_info.get_attribute('title')
                link = list_info.get_attribute('href')
                
                options = block.find_element(By.CLASS_NAME, 'option').find_elements(By.TAG_NAME, 'span')
                options = ','.join([option.text for option in options])

                detail_fields = block.find_element(By.CLASS_NAME, 'etc').text

                data = {'회사명' : company_name, '채용공고 제목' : title, '채용공고 세부 사항' : options, '기술 세부 사항' : detail_fields, '링크' : link}
                current_page_data.append(data)

            except NoSuchElementException as e:
                print(f'\n\nBlock No : {i}, \n\t{e}')

        return current_page_data
    

    def crawling(self):
        """crawling을 수행하는 주요 함수."""
        browser = webdriver.Chrome(options=self.options)
        browser.get(self.url)
        self.input_keyword(browser)

        idx = 1
        total_data = []
        while True:
            if self.debug and idx == 5:
                    break

            if idx > 1:
                url = f'https://www.jobkorea.co.kr/Search/?stext=%EB%94%A5%EB%9F%AC%EB%8B%9D&tabType=recruit&Page_No={idx}'
                try:
                    browser.get(url)
                except:
                    print(f'Page Not Found. Current Page No : {idx}')
                    break
        
            time.sleep(self.wait_sec)
            page_data = self.get_data_from_page(browser, idx)
            total_data.extend(page_data)
                    
            idx += 1

        return total_data