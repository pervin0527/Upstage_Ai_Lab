import pandas as pd

from src.wanted import WantedCrawler
from src.saramin import SaraminCrawler
from src.jobkorea import JobKoreaCrawler

from utils.util import save_dataset


def main():
    ## 잡코리아 크롤링
    # jobkorea_crawler = JobKoreaCrawler(KEYWORD, WAIT_SEC, DEBUG)
    # jobkorea_dataset = jobkorea_crawler.crawling()
    # save_dataset(f"{SAVE_PATH}", "job_korea", jobkorea_dataset)

    ## 사람인 크롤링
    # saramin_crawler = SaraminCrawler(KEYWORD, WAIT_SEC, DEBUG)
    # saramin_dataset = saramin_crawler.crawling()
    # save_dataset(f"{SAVE_PATH}", "saramin", saramin_dataset)

    ## 원티드 크롤링
    wanted_crawler = WantedCrawler(KEYWORD, WAIT_SEC, DEBUG)
    wanted_dataset = wanted_crawler.crawling()
    save_dataset(f"{SAVE_PATH}", "wanted", wanted_dataset)
    


if __name__ == "__main__":
    SAVE_PATH = "./outputs"
    KEYWORD = '딥러닝'
    WAIT_SEC = 3
    DEBUG = True

    main()