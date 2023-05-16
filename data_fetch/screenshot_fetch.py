import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import random


class Browser:
    browser, service = None, None

    # Initialise the webdriver with the path to chromedriver.exe
    def __init__(self, driver: str):
        self.service = Service(driver)
        self.browser = webdriver.Chrome(service=self.service)

    def open_page(self, url: str):
        self.browser.get(url)

    def close_browser(self):
        self.browser.close()

    def add_input(self, by: By, value: str, text: str):
        field = self.browser.find_element(by=by, value=value)
        field.send_keys(text)
        time.sleep(1)

    def click_button(self, by: By, value: str):
        button = self.browser.find_element(by=by, value=value)
        button.click()
        time.sleep(1)

    def login_collegerama(self, username: str, password: str):
        self.add_input(by=By.ID, value='username', text=username)
        time.sleep(random.randint(1, 3))
        self.add_input(by=By.ID, value='password', text=password)
        time.sleep(random.randint(1, 3))
        self.click_button(by=By.CLASS_NAME, value='login-button')


if __name__ == '__main__':
    
    with open('./secrets.txt', 'r') as f:
        username, password = f.read().splitlines()
    
    browser = Browser('/usr/bin/chromedriver')

    browser.open_page('https://collegeramacolleges.tudelft.nl/new/portfolio-items/eemcs-msc-cs/')
    time.sleep(random.randint(4, 10))
    

    browser.login_collegerama(username=username, password=password)
    time.sleep(10)

    browser.close_browser()