import time

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

class SaraminCrawler:
    def __init__(self, wait_sec=3, debug=True):
        self.url = 'https://www.saramin.co.kr/zf_user/'
        self.wait_sec = wait_sec

        self.debug = debug
        self.options = Options()
        if debug:
            self.options.add_experimental_option("detach", True)


    def input_setup(self, browser):
        """검색조건 설정"""
        browser.find_element(By.CLASS_NAME, 'recruit').click()
        seoul_all = browser.find_element(By.XPATH, '//*[@id="loc_mcd_101000"]') ## 서울 전체
        browser.execute_script("arguments[0].click();", seoul_all)

        gyeonggi_all = browser.find_element(By.XPATH, '//*[@id="depth1_btn_102000"]/button') ## 경기
        browser.execute_script("arguments[0].click();", gyeonggi_all)

        browser.find_element(By.CLASS_NAME, 'job_category_section').click()
        browser.find_element(By.XPATH, '//*[@id="sp_main_wrapper"]/div[2]/div/div[2]/div[2]/div[1]/button[6]').click()
        browser.find_element(By.XPATH, '//*[@id="sp_job_category_subDepth_2"]/div[2]/div/div[2]/div/dl[2]/dd/button[6]').click()
        browser.find_element(By.XPATH, '//*[@id="sp_job_category_subDepth_2"]/div[2]/div/div[2]/div/dl[2]/dd/button[7]').click()
        browser.find_element(By.XPATH, '//*[@id="search_btn"]').click()


    def get_data_from_page(self, browser, page_number):
        container = browser.find_element(By.CLASS_NAME, 'list_body')
        blocks = container.find_elements(By.CLASS_NAME, 'list_item')  

        if len(blocks) < 30:
            return False

        current_page_data = []
        for i, block in tqdm(enumerate(blocks), desc=f"Page {page_number}", total=len(blocks), leave=True):
            try:
                block.get_attribute('id')
                company_name_box = block.find_element(By.CLASS_NAME, 'company_nm')
                company_name = company_name_box.find_element(By.CLASS_NAME, 'str_tit').text

                recurit_box = block.find_element(By.CLASS_NAME, 'notification_info')
                title_box = recurit_box.find_element(By.CLASS_NAME, 'str_tit')
                title = title_box.text
                link = title_box.get_attribute('href')

                detail_fields_box = block.find_element(By.CLASS_NAME, 'job_meta').find_element(By.CLASS_NAME, 'job_sector')
                detail_fields = detail_fields_box.find_elements(By.TAG_NAME, 'span')
                detail_fields = ','.join([x.text for x in detail_fields])

                option_box = block.find_element(By.CLASS_NAME, 'recruit_info')
                options = option_box.find_elements(By.TAG_NAME, 'p')
                options = ','.join([option.text.replace(' · ', '') for option in options])

                data = {'회사명' : company_name, '채용공고 제목' : title, '채용공고 세부 사항' : options, '기술 세부 사항' : detail_fields, '링크' : link}
                current_page_data.append(data)
                
            except:
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
                url = f'https://www.saramin.co.kr/zf_user/jobs/list/domestic?page={idx}&loc_mcd=101000%2C102000&cat_kewd=108%2C109&search_optional_item=n&search_done=y&panel_count=y&preview=y&isAjaxRequest=0&page_count=50&sort=RL&type=domestic&is_param=1&isSearchResultEmpty=1&isSectionHome=0&searchParamCount=2#searchTitle'
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