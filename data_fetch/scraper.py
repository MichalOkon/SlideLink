import os
from typing import Tuple
from tqdm import tqdm
from browser import Browser
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException


class Scraper:
    """Scraper class"""

    def __init__(self, link: str, browser: Browser):
        """Initialise scraper object.

        Args:
            link (str): resource to scrape from
        """
        self.browser = browser
        self.total_time = 0
        self.curr_time = 0
        self.slideshow_count = 0
        self.presenter_count = 0
        self.curr_timestamp = 0
        self.link = link
        self.lecture_name = link.split("/")[-1]

    def login_action(self, username: str, password: str) -> None:
        """Perform login action with username and password.

        Args:
            username (str): username
            password (str): password
        """
        self.browser.add_text_input(by=By.ID, value="username", text=username)
        self.browser.add_text_input(by=By.ID, value="password", text=password)
        self.browser.click_button(by=By.CLASS_NAME, value="login-button")

    def take_screen_shot(self, screen_type: str, lecture_name: str) -> None:
        """Takes screen shots and saves them to directories.

        Args:
            screen_type (str): screen type to save screenshots from.
            lecture_name (str): _description_
        """
        # Create the lecture folder if it does not exist
        if not os.path.exists(f"screenshots/{lecture_name}"):
            os.makedirs(f"screenshots/{lecture_name}")
        if not os.path.exists(f"screenshots/{lecture_name}/{screen_type}"):
            os.makedirs(f"screenshots/{lecture_name}/{screen_type}")

        # Take the screenshot
        count = 0
        if screen_type == "slideshow":
            self.slideshow_count += 1
            count = self.slideshow_count
        elif screen_type == "presenter":
            self.presenter_count += 1
            count = self.presenter_count
        save_path = f"screenshots/{lecture_name}/{screen_type}/{count:04}.png"
        self.browser.screenshot(save_path)

    def move_next_timestamp(self, times: int, wait: bool = True) -> None:
        """Moves the next timestamp of the video on Collegerama to through right arrow key.

        Args:
            times (int): number of presses
            wait (bool, optional): defines if one should wait for element to be found. Defaults to True.
        """
        for _ in range(times):
            if wait:
                video = self.browser._wait_until_present(By.ID, "vjs_video_3")
                if video is None:
                    continue
            else:
                video = self.browser.find_element(By.ID, value="vjs_video_3")
            video.send_keys(Keys.ARROW_RIGHT)
        self.browser.sleep(1, 3)

    def get_video_times_m(self, wait: bool = True) -> Tuple[int, int]:
        """Start counting the time (in minutes)

        Returns:
            Tuple[int, int]: start and end times tuple in minutes
        """
        self.browser.sleep(1, 4)
        string_curr_time = None
        string_total_time = None
        if wait:
            string_curr_time = self.browser._wait_until_click(By.CSS_SELECTOR, value=".vjs-current-time-display")
            string_curr_time = string_curr_time.text if string_curr_time is not None else None
            string_total_time = self.browser._wait_until_click(By.CSS_SELECTOR, value=".vjs-duration-display")
            string_total_time = string_total_time.text if string_total_time is not None else None
        if string_curr_time is None:
            string_curr_time = self.browser.find_element(By.CSS_SELECTOR, value=".vjs-current-time-display").text
        if string_total_time is None:
            string_total_time = self.browser.find_element(By.CSS_SELECTOR, value=".vjs-duration-display").text

        times = [string_curr_time, string_total_time]
        times_s = [0 for _ in range(2)]

        for i in range(len(times)):
            # Turn time into seconds
            video_time_strs = times[i].split(":")

            # Turn strings into integers
            if len(video_time_strs) == 3:
                times_s[i] = 60 * int(video_time_strs[0]) + int(video_time_strs[1])
            else:
                times_s[i] = int(video_time_strs[0])

        return tuple(times_s)

    def scrape_lecture_recording(self, video_url: str):
        """Scrapes screenshots of a video recording.

        Args:
            video_url (str): video URL to scrape
        """
        with open("./secrets.txt", "r") as f:
            username, password = f.read().splitlines()

        self.browser.driver.maximize_window()
        self.browser.open_page(video_url)
        self.browser.sleep(0.5, 1.5)

        self.login_action(username=username, password=password)
        self.browser.sleep(5, 7)

        # Turn on the fullscreen
        self.browser.click_button(By.CSS_SELECTOR, ".btn-bigger.link.channel-button")
        self.browser.sleep(0.5, 1.5)

        self.browser.click_button(By.XPATH, '//*[@id="InAppPlayerContainer"]/div[2]/div[1]')
        self.browser.sleep(0.5, 1.5)

        # Enter the media player
        self.browser.iframe_switch("player-iframe")
        self.browser.sleep(0.5, 1.5)

        self.browser.click_execute(By.XPATH, '//*[@id="vjs_video_3"]/div[4]/button[1]')
        self.browser.sleep(0.5, 1.5)

        children = self.browser.get_children(By.XPATH, '//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]')
        children = children if children is not None else []
        if len(children) == 2:
            slide_show_path = '//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[1]'
            presenter_path = '//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[2]'
        else:
            slide_show_path = '//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[2]'
            presenter_path = '//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[3]'
        self.browser.sleep(0.5, 1.5)
        self.browser.click_execute(By.XPATH, '//*[@id="vjs_video_3"]/div[4]/button[1]', wait=False)
        self.browser.sleep(3.5, 4)

        # Get total time of the video (in minutes)
        curr_time, total_time = self.get_video_times_m()
        self.total_time = total_time

        self.browser.sleep(0.5, 1.5)

        self.curr_time = curr_time

        print(f"Scraping video: {self.link}")

        progress_bar = tqdm(desc="Progress: ", total=self.total_time, colour="red")
        progress_bar.update(self.curr_time)

        while self.curr_time < self.total_time:
            try:
                self.browser.click_button(By.XPATH, slide_show_path, wait=False)
                self.browser.sleep(0.5, 2.5)

                self.take_screen_shot("slideshow", self.lecture_name)
                self.browser.sleep(0.5, 2.5)

                self.browser.click_execute(By.CLASS_NAME, "mediasite-player__smart-zoom-exit-button", wait=False)
                self.browser.sleep(0.5, 2.5)

                self.browser.click_button(By.XPATH, presenter_path, wait=False)
                self.browser.sleep(0.5, 2.5)

                self.take_screen_shot("presenter", self.lecture_name)
                self.browser.sleep(0.5, 2.5)

                self.browser.click_execute(By.CLASS_NAME, "mediasite-player__smart-zoom-exit-button", wait=False)
                self.browser.sleep(0.5, 2.5)

                self.move_next_timestamp(12)
                curr_time, _ = self.get_video_times_m()
                time_diff = curr_time - self.curr_time
                self.curr_time = curr_time
                progress_bar.update(time_diff)
            except TimeoutException:
                progress_bar.update(self.total_time - self.curr_time)
                break
        progress_bar.close()
        print("Scraping DONE")
        self.browser.close_browser()
