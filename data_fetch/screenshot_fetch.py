import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import random


class Browser:
    browser, service = None, None

    # Initialise the webdriver with the path to chromedriver.exe
    def __init__(self, driver: str):
        self.service = Service(driver)
        self.browser = webdriver.Chrome(service=self.service)
        self.count = 1

    def open_page(self, url: str):
        self.browser.get(url)
    
    def sleep(self, start_range: int, end_range: int):
        time.sleep(random.randint(start_range, end_range))

    def close_browser(self):
        self.browser.close()

    def add_input(self, by: By, value: str, text: str):
        field = self.browser.find_element(by=by, value=value)
        field.send_keys(text)
        self.sleep(1, 4)

    def click_button(self, by: By, value: str):
        button = self.browser.find_element(by=by, value=value)
        button.click()
        self.sleep(1, 4)

    def click_execute(self, by: By, value):
        button = self.browser.find_element(by=by, value=value)
        self.browser.execute_script("arguments[0].click();", button)
        

    def login_collegerama(self, username: str, password: str):
        self.add_input(by=By.ID, value='username', text=username)
        self.add_input(by=By.ID, value='password', text=password)
        self.click_button(by=By.CLASS_NAME, value='login-button')
    
    def iframe_switch(self, value: str):
        self.browser.switch_to.frame(value)
    
    def screen_shot(self, name: str):
        self.browser.save_screenshot(f"screenshots/{name}_{self.count}.png")
        self.count += 1

if __name__ == '__main__':
    
    with open('./secrets.txt', 'r') as f:
        username, password = f.read().splitlines()
    
    browser = Browser('/usr/bin/chromedriver')

    browser.open_page('https://collegerama.tudelft.nl/Mediasite/Channel/eemcs-msc-cs/watch/770911dbcaad427990eceb9afe0e01db1d')
    browser.sleep(1, 3)
    

    browser.login_collegerama(username=username, password=password)
    browser.sleep(5, 7)
    
    browser.click_button(By.XPATH, value='//*[@id="InAppPlayerContainer"]/div[2]/div[1]')
    browser.sleep(1, 3)


    # ENTER the media player
    browser.iframe_switch("player-iframe")
    browser.sleep(1, 3)
    
    
    browser.click_execute(By.XPATH, value='//*[@id="vjs_video_3"]/div[4]/button[1]')
    browser.sleep(1, 3)
    
    browser.click_button(By.XPATH, value='//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[2]')
    browser.sleep(1, 3)
    
    browser.screen_shot("slideshow")
    browser.sleep(1, 3)
    
    browser.click_button(By.CLASS_NAME, value="mediasite-player__smart-zoom-exit-button")
    browser.sleep(1, 3)
    
    browser.click_button(By.XPATH, value='//*[@id="vjs_video_3"]/div[7]/div[10]/div[1]/div[3]')
    browser.sleep(1, 3)
    
    browser.screen_shot("presenter")
    browser.sleep(1, 3)
    
    browser.click_button(By.CLASS_NAME, value="mediasite-player__smart-zoom-exit-button")
    browser.sleep(1, 3)

    browser.close_browser()

