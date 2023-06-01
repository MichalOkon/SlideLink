from scraper import Scraper
from browser import Browser

with open("./urls.txt", "r") as f:
    urls = f.read().splitlines()

browser = Browser("/usr/bin/chromedriver")

for url in urls:
    scraper = Scraper(url, browser)
    scraper.scrape_lecture_recording(url)
