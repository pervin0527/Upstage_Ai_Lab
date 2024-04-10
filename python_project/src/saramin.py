import time

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

class SaraminCrawler:
    def __init__(self, keyword, wait_sec=3, debug=True):
        self.url = 'https://www.saramin.co.kr/zf_user/'
        self.keyword = keyword
        self.wait_sec = wait_sec

        self.debug = debug
        self.options = Options()
        if debug:
            self.options.add_experimental_option("detach", True)


    def input_keyword(self, browser):
        """검색창에 keyword를 입력"""
        browser.find_element(By.CLASS_NAME, 'btn_search').click()
        keyword_input = browser.find_element(By.ID, 'ipt_keyword_recruit')
        keyword_input.click()
        keyword_input.send_keys(self.keyword)

        btn_search_recruit = browser.find_element(By.ID, 'btn_search_recruit')
        btn_search_recruit.click()

        browser.find_element(By.XPATH, '//*[@id="content"]/ul[1]/li[2]/a').click()


    def get_data_from_page(self, browser, page_number):
        container = browser.find_element(By.CLASS_NAME, 'content')
        blocks = container.find_elements(By.CLASS_NAME, 'item_recruit')        

        current_page_data = []
        for i, block in tqdm(enumerate(blocks), desc=f"Page {page_number}", total=len(blocks), leave=False):
            try:
                company_name = block.find_element(By.CLASS_NAME, 'corp_name').find_element(By.TAG_NAME, 'a').text
                job_tit = block.find_element(By.CLASS_NAME, 'job_tit').find_element(By.TAG_NAME, 'a')
                title = job_tit.get_attribute('title')
                link = job_tit.get_attribute('href')
                
                options = []
                job_condition = block.find_element(By.CLASS_NAME, 'job_condition').find_elements(By.TAG_NAME, 'span')
                for j, condition in enumerate(job_condition):
                    if j == 0:
                        regions = condition.find_elements(By.TAG_NAME, 'a')
                        regions = ','.join([region.text for region in regions])
                        options.append(regions)
                    else:
                        options.append(condition.text)
                options = ','.join([option for option in options])

                detail_fields = block.find_element(By.CLASS_NAME, 'job_sector').find_elements(By.TAG_NAME, 'a')
                detail_fields = ','.join([x.text for x in detail_fields])

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
                url = f'https://www.saramin.co.kr/zf_user/search/recruit?search_area=main&search_done=y&search_optional_item=n&searchType=search&searchword=%EB%94%A5%EB%9F%AC%EB%8B%9D&recruitPage={idx}&recruitSort=relation&recruitPageCount=40&inner_com_type=&company_cd=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7%2C9%2C10&show_applied=&quick_apply=&except_read=&ai_head_hunting='
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