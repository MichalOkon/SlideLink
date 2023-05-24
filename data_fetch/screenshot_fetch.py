import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import random


class Browser:

    # Initialise the webdriver with the path to chromedriver.exe
    def __init__(self, driver: str):
        self.service = Service(driver)
        self.driver = webdriver.Chrome(service=self.service)
        self.slideshow_count = 1
        self.presenter_count = 1
        self.curr_timestamp = 0

    def open_page(self, url: str):
        self.driver.get(url)

    def sleep(self, start_range: int, end_range: int):
        time.sleep(random.randint(start_range, end_range))

    def close_browser(self):
        self.driver.close()

    def add_input(self, by: By, value: str, text: str):
        field = self.driver.find_element(by=by, value=value)
        field.send_keys(text)
        self.sleep(1, 4)

    def click_button(self, by: By, value: str):
        button = self.driver.find_element(by=by, value=value)
        button.click()
        self.sleep(1, 4)

    def click_execute(self, by: By, value):
        button = self.driver.find_element(by=by, value=value)
        self.driver.execute_script("arguments[0].click();", button)

    def login_collegerama(self, username: str, password: str):
        self.add_input(by=By.ID, value='username', text=username)
        self.add_input(by=By.ID, value='password', text=password)
        self.click_button(by=By.CLASS_NAME, value='login-button')

    def iframe_switch(self, value: str):
        self.driver.switch_to.frame(value)

    def screen_shot(self, name: str, ss_type: str):
        if ss_type == "slideshow":
            self.driver.save_screenshot(f"screenshots/{name}_{self.slideshow_count}.png")
            self.slideshow_count += 1
        elif ss_type == "presenter":
            self.driver.save_screenshot(f"screenshots/{name}_{self.presenter_count}.png")
            self.presenter_count += 1

    def next_timestamp(self, times):
        # Click the right arrow button times number of times
        video = self.driver.find_element(By.ID, value="vjs_video_3")
        for i in range(times):
            video.send_keys(Keys.ARROW_RIGHT)
        self.sleep(1, 3)

    def get_total_video_time(self):
        string_time = self.driver.find_element(By.CSS_SELECTOR, value=".vjs-duration-display").text
        print(string_time)
        # Turn time into seconds
        video_time = string_time.split(":")
        print(video_time)

        # Turn strings into integers
        if (len(video_time) == 3):
            total_time = 60 * int(video_time[0]) + int(video_time[1])
        else:
            total_time = int(video_time[0])

        return total_time

# class Scraper:
#
#         def __init__(self):
#             self.total_time = 0
#             self.curr_time = 0
#             self.browser = Browser('/usr/bin/chromedriver')
#

if __name__ == '__main__':

    with open('./secrets.txt', 'r') as f:
        username, password = f.read().splitlines()

    browser = Browser('/usr/bin/chromedriver')
    browser.driver.maximize_window()
    browser.open_page(
        'https://collegerama.tudelft.nl/Mediasite/Channel/eemcs-msc-cs/watch/770911dbcaad427990eceb9afe0e01db1d')
    browser.sleep(1, 3)

    browser.login_collegerama(username=username, password=password)
    browser.sleep(5, 7)

    # Turn on the fullscreen
    browser.click_button(By.CSS_SELECTOR, value='.btn-bigger.link.channel-button')
    browser.sleep(1, 3)

    browser.click_button(By.XPATH, value='//*[@id="InAppPlayerContainer"]/div[2]/div[1]')
    browser.sleep(1, 3)

    # Enter the media player
    browser.iframe_switch("player-iframe")
    browser.sleep(1, 3)

    browser.click_execute(By.XPATH, value='//*[@id="vjs_video_3"]/div[4]/button[1]')
    browser.sleep(1, 3)

    # Get total time of the video (in minutes)
    total_time = browser.get_total_video_time()

    browser.sleep(1, 3)

    # Start counting the time (in minutes)
    curr_time = 0

    while curr_time < total_time:
        browser.click_button(By.XPATH, value='//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[2]')
        browser.sleep(1, 3)

        browser.screen_shot("slideshow", "slideshow")
        browser.sleep(1, 3)

        browser.click_button(By.CLASS_NAME, value="mediasite-player__smart-zoom-exit-button")
        browser.sleep(1, 3)

        browser.click_button(By.XPATH, value='//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[3]')
        browser.sleep(1, 3)

        browser.screen_shot("presenter", "presenter")
        browser.sleep(1, 3)

        browser.click_button(By.CLASS_NAME, value="mediasite-player__smart-zoom-exit-button")
        browser.sleep(1, 3)

        browser.next_timestamp(6)
        curr_time += 1
    browser.close_browser()
