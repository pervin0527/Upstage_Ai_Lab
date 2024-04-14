import re
import time

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

class WantedCrawler:
    def __init__(self, wait_sec=3, total_scroll=100, debug=True):
        self.url = 'https://www.wanted.co.kr/'
        self.wait_sec = wait_sec
        self.total_scroll = total_scroll

        self.debug = debug
        self.options = Options()
        if debug:
            self.options.add_experimental_option("detach", True)


    def input_setup(self, browser):
        """검색조건 설정"""
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

        idx = 0  # For Debug mode
        last_height = browser.execute_script("return document.body.scrollHeight")

        with tqdm(total=self.total_scroll) as pbar:
            while idx < self.total_scroll:
                if self.debug and idx == 1:
                    break
                browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = browser.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                idx += 1  # For Debug mode
                pbar.update(1)


    def crawling(self):
        """crawling을 수행하는 주요 함수."""
        browser = webdriver.Chrome(options=self.options)
        browser.get(self.url)
        self.input_setup(browser)

        self.scroll_down(browser)
        container = browser.find_element(By.CLASS_NAME, 'List_List__FsLch')
        blocks = container.find_elements(By.CLASS_NAME, 'JobCard_JobCard__oZL4d')

        total_data = []
        for i, block in tqdm(enumerate(blocks), total=len(blocks), leave=True):
            try:
                card = block.find_element(By.TAG_NAME, 'a')
                company_name = card.get_attribute('data-company-name')
                title = card.get_attribute('data-position-name')
                link = card.get_attribute('href')

                card_browser = webdriver.Chrome()
                card_browser.get(link)
                time.sleep(1.5)

                option_block = card_browser.find_element(By.CLASS_NAME, 'JobHeader_JobHeader__Tools__n5Vcg')
                options = option_block.find_elements(By.CLASS_NAME, 'JobHeader_JobHeader__Tools__Company__Info__omnQX')
                option_list = [x.text for x in options]
                option_list = ','.join(option_list)

                card_browser.find_element(By.CLASS_NAME, 'Button_Button__outlinedSizeLarge__n_OOf').click() ## 상세 정보 더 보기 버튼
                div_blocks = card_browser.find_elements(By.CLASS_NAME, 'JobDescription_JobDescription__paragraph__Iwfqn')
                
                detail_fields = []
                for db in div_blocks:
                    db_title = db.find_element(By.TAG_NAME, 'h3').text
                    
                    if db_title in ['주요업무', '자격요건', '우대사항']:
                        db_li = db.find_elements(By.TAG_NAME, 'span')
                        # db_li = [re.sub(r'[^가-힣A-Za-z0-9\s]', '', x.text).replace('\n', ',') for x in db_li]
                        # db_li = [x.text.replace('\n', ',') for x in db_li]
                        
                        db_li = [re.sub('<br>', '', x.get_attribute('innerHTML')).replace('\n', ',').strip() for x in db_li]
                        db_li = [re.sub(r'<[^>]+>', '', x) for x in db_li]
                        db_li = [x.replace("• ", ",").replace('ㆍ', ",") for x in db_li][0:]
                        detail_fields.extend(db_li)
                detail_fields = ','.join(detail_fields)
                
                data = {'회사명' : company_name, '채용공고 제목' : title, '채용공고 세부 사항' : option_list, '기술 세부 사항' : detail_fields, '링크' : link}
                card_browser.close()
                total_data.append(data)

            except NoSuchElementException as e:
                continue

        browser.close()
        return total_data