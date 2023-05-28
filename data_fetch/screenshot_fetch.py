import os
import time
import random
from tqdm import tqdm
from typing import Tuple
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException


IGNORE_EXCEPTIONS=(NoSuchElementException,StaleElementReferenceException)

class Browser:

    # Initialise the webdriver with the path to chromedriver.exe
    def __init__(self, driver: str):
        self.service = Service(driver)
        self.driver = Chrome(service=self.service)
        self.driver.delete_all_cookies()
        self.slideshow_count = 0
        self.presenter_count = 0
        self.curr_timestamp = 0
    
    def _wait_until_click(self, by: By, value: str, timeout: float = 20.0):
        return WebDriverWait(
            self.driver,
            timeout=timeout,
            ignored_exceptions=IGNORE_EXCEPTIONS
            ).until(
                ec.element_to_be_clickable(
                    (by, value)
                )
            )
    
    def _wait_until_present(self, by: By, value: str, timeout: float = 20.0):
        return WebDriverWait(
            self.driver,
            timeout=timeout,
            ignored_exceptions=IGNORE_EXCEPTIONS
            ).until(
                ec.presence_of_element_located(
                    (by, value)
                )
            )

    def _wait_until_selectable(self, by: By, value: str, timeout: float = 20.0):
        return WebDriverWait(
            self.driver,
            timeout=timeout,
            ignored_exceptions=IGNORE_EXCEPTIONS
            ).until(
                ec.element_to_be_selected(
                    (by, value)
                )
            )

    def open_page(self, url: str) -> None:
        self.driver.get(url)

    def sleep(self, start_range: float, end_range: float) -> None:
        time.sleep(random.uniform(start_range, end_range))

    def close_browser(self) -> None:
        self.driver.close()

    def add_input(self, by: By, value: str, text: str, wait: bool = True) -> None:
        if wait :
            field = self._wait_until_present(by, value)
        else :
            field = self.driver.find_element(by=by, value=value)
        field.send_keys(text)

    def click_button(self, by: By, value: str, wait: bool = True) -> None:
        if wait :
            button = self._wait_until_click(by, value)
        else :
            button = self.driver.find_element(by=by, value=value)
        button.click()

    def click_execute(self, by: By, value: str, wait: bool = True) -> None:
        if wait :
            button = self._wait_until_present(by, value)
        else :
            button = self.driver.find_element(by=by, value=value)
        self.driver.execute_script("arguments[0].click();", button)

    def login_collegerama(self, username: str, password: str) -> None:
        self.add_input(by=By.ID, value='username', text=username)
        self.add_input(by=By.ID, value='password', text=password)
        self.click_button(by=By.CLASS_NAME, value='login-button')

    def iframe_switch(self, value: str) -> None:
        self.driver.switch_to.frame(value)

    def screen_shot(self, ss_type: str, lecture_name: str) -> None:
        # Create the lecture folder if it does not exist
        if not os.path.exists(f"screenshots/{lecture_name}"):
            os.makedirs(f"screenshots/{lecture_name}")
        if not os.path.exists(f"screenshots/{lecture_name}/{ss_type}"):
            os.makedirs(f"screenshots/{lecture_name}/{ss_type}")

        # Take the screenshot
        if ss_type == "slideshow":
            self.slideshow_count += 1
            count = self.slideshow_count
        elif ss_type == "presenter":
            self.presenter_count += 1
            count = self.presenter_count
        self.driver.save_screenshot(f"screenshots/{lecture_name}/{ss_type}/{count:04}.png")

    def next_timestamp(self, times: int, wait: bool = True):
        for _ in range(times):
            if wait :
                video = self._wait_until_present(By.ID, "vjs_video_3")
            else :
                video = self.driver.find_element(By.ID, value="vjs_video_3")
            video.send_keys(Keys.ARROW_RIGHT)
        self.sleep(1, 3)
    
    def get_video_times_m(self, wait: bool = True) -> Tuple[int, int]:
        """Start counting the time (in minutes)

        Returns:
            tuple(int, int): start and end times tuple in minutes
        """
        self.sleep(1, 4)
        if wait :
            string_curr_time = self._wait_until_click(By.CSS_SELECTOR, value=".vjs-current-time-display").text
            string_total_time = self._wait_until_click(By.CSS_SELECTOR, value=".vjs-duration-display").text
        else :
            string_curr_time = self.driver.find_element(By.CSS_SELECTOR, value=".vjs-current-time-display").text
            string_total_time = self.driver.find_element(By.CSS_SELECTOR, value=".vjs-duration-display").text
        
        times = [string_curr_time, string_total_time]

        for i in range(len(times)):
        # Turn time into seconds
            video_time_strs = times[i].split(":")

            # Turn strings into integers
            if (len(video_time_strs) == 3):
                times[i] = 60 * int(video_time_strs[0]) + int(video_time_strs[1])
            else:
                times[i] = int(video_time_strs[0])

        return tuple(times)
    
    def get_children(self, by: By, value: str, wait: bool = True):
        if wait :
            element = self._wait_until_present(by, value)
        else :
            element = self.driver.find_element(by, value=value)
        return element.find_elements(By.TAG_NAME, "div")

class Scraper:

        def __init__(self, link: str):
            self.total_time = 0
            self.curr_time = 0
            self.link = link
            self.lecture_name = link.split("/")[-1]

        def scrape_lecture_recording(self, video_url: str):
            with open('./secrets.txt', 'r') as f:
                username, password = f.read().splitlines()

            browser = Browser('/usr/bin/chromedriver')
            browser.driver.maximize_window()
            browser.open_page(video_url)
            browser.sleep(0.5, 1.5)

            browser.login_collegerama(username=username, password=password)
            browser.sleep(5, 7)

            # Turn on the fullscreen
            browser.click_button(By.CSS_SELECTOR, '.btn-bigger.link.channel-button')
            browser.sleep(0.5, 1.5)

            browser.click_button(By.XPATH, '//*[@id="InAppPlayerContainer"]/div[2]/div[1]')
            browser.sleep(0.5, 1.5)

            # Enter the media player
            browser.iframe_switch("player-iframe")
            browser.sleep(0.5, 1.5)

            browser.click_execute(By.XPATH, '//*[@id="vjs_video_3"]/div[4]/button[1]')
            browser.sleep(0.5, 1.5)
            
            children = browser.get_children(By.XPATH, '//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]')
            if len(children) == 2:
                slide_show_path = '//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[1]'
                presenter_path = '//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[2]'
            else:
                slide_show_path = '//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[2]'
                presenter_path = '//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[3]'
            browser.sleep(0.5, 1.5)
            browser.click_execute(By.XPATH, '//*[@id="vjs_video_3"]/div[4]/button[1]', wait=False)
            browser.sleep(3.5, 4)

            # Get total time of the video (in minutes)
            curr_time, total_time  = browser.get_video_times_m(wait=False)
            self.total_time = total_time

            browser.sleep(0.5, 1.5)

            self.curr_time = curr_time
            
            print(f"Scraping video: {self.link}")
            
            progress_bar = tqdm(desc="Progress: ", total=self.total_time, colour="red")
            progress_bar.update(self.curr_time)

            while self.curr_time < self.total_time :
                try: 
                    browser.click_button(By.XPATH, slide_show_path, wait=False)
                    browser.sleep(0.5, 2.5)

                    browser.screen_shot("slideshow", self.lecture_name)
                    browser.sleep(0.5, 2.5)

                    browser.click_execute(By.CLASS_NAME, "mediasite-player__smart-zoom-exit-button", wait=False)
                    browser.sleep(0.5, 2.5)

                    browser.click_button(By.XPATH, presenter_path, wait=False)
                    browser.sleep(0.5, 2.5)

                    browser.screen_shot("presenter", self.lecture_name)
                    browser.sleep(0.5, 2.5)

                    browser.click_execute(By.CLASS_NAME, "mediasite-player__smart-zoom-exit-button", wait=False)
                    browser.sleep(0.5, 2.5)

                    browser.next_timestamp(12)
                    curr_time, _  = browser.get_video_times_m()
                    time_diff = curr_time - self.curr_time
                    self.curr_time = curr_time
                    progress_bar.update(time_diff)
                except TimeoutException:
                    progress_bar.update(self.total_time - self.curr_time)
                    break
            progress_bar.close()
            print("Scraping DONE")
            browser.close_browser()


if __name__ == '__main__':
    # Get the url of the recording from the command line
    with open('./urls.txt', 'r') as f:
        urls = f.read().splitlines()

    for url in urls:
        scraper = Scraper(url)
        scraper.scrape_lecture_recording(url)
