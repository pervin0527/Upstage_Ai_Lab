import pandas as pd
from src.job_korea import JobKoreaCrawler
from utils.util import save_dataset


def main():
    job_korea_crawler = JobKoreaCrawler(KEYWORD, WAIT_SEC, DEBUG)
    job_korea_dataset = job_korea_crawler.crawling()
    save_dataset(f"{SAVE_PATH}", "job_korea", job_korea_dataset)


if __name__ == "__main__":
    SAVE_PATH = "./outputs"
    KEYWORD = '딥러닝'

    WAIT_SEC = 3
    DEBUG = True

    main()