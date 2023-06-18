from scraper import Scraper
from browser import Browser

browser = Browser("/usr/bin/chromedriver")

with open("./urls.txt", "r") as f:
    urls_with_indexes = f.read().splitlines()

for url_with_indexes in urls_with_indexes:
    url, slides_view_index, presenter_viewer_index = url_with_indexes
    scraper = Scraper(
        url, int(slides_view_index), int(presenter_viewer_index), browser
    )
    scraper.scrape_lecture_recording(url)
