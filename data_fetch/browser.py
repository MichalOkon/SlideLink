import time
import random
from typing import Union, Any
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException


IGNORE_EXCEPTIONS = (NoSuchElementException, StaleElementReferenceException)


class Browser:
    """Browser driver abstraction capabilities class."""

    def __init__(self, driver_path: str):
        """Initialise the webdriver with the path to the chromedriver.

        Args:
            driver_path (str): path to the chromedriver
        """
        self.driver = Chrome(service=Service(driver_path))
        self.driver.delete_all_cookies()

    def _wait_until_click(self, by: str, value: str, timeout: float = 20.0) -> Union[Any, None]:
        """Waits until a specified element can be clicked.

        Args:
            by (str): Identifier of the element type
            value (str): value of the identifier
            timeout (float, optional): A timeout value. Defaults to 20.0.

        Returns:
            Union[Any, None]: A Web Element or none if
        """
        try:
            clickable_el = WebDriverWait(self.driver, timeout=timeout, ignored_exceptions=IGNORE_EXCEPTIONS).until(
                ec.element_to_be_clickable((by, value))
            )
        except TimeoutException:
            clickable_el = None
        return clickable_el

    def _wait_until_present(self, by: str, value: str, timeout: float = 20.0) -> Union[Any, None]:
        """Waits until a specified element is present (can be located).

        Args:
            by (str): Identifier of the element type
            value (str): value of the identifier
            timeout (float, optional): A timeout value. Defaults to 20.0.

        Returns:
            Union[Any, None]: A Web Element or none if
        """
        try:
            present_el = WebDriverWait(self.driver, timeout=timeout, ignored_exceptions=IGNORE_EXCEPTIONS).until(
                ec.presence_of_element_located((by, value))
            )
        except TimeoutException:
            present_el = None
        return present_el

    def _wait_until_found(self, by: str, value: str, timeout: float = 20.0) -> Union[Any, None]:
        """Waits until a specified element is found through active element finding.

        Args:
            by (str): Identifier of the element type
            value (str): value of the identifier
            timeout (float, optional): A timeout value. Defaults to 20.0.

        Returns:
            Union[Any, None]: A Web Element or none if
        """
        return WebDriverWait(self.driver, timeout=timeout, ignored_exceptions=IGNORE_EXCEPTIONS).until(
            self.driver.find_element(by=by, value=value)
        )

    def open_page(self, url: str) -> None:
        """Open a new page with a given url.

        Args:
            url (str): resource to open.
        """
        self.driver.get(url)

    def sleep(self, start_range: float, end_range: float) -> None:
        """Wrapper method for the driver to sleep for some duration range in seconds.

        Args:
            start_range (float): start time range in seconds
            end_range (float): end time range in seconds (exclusive)
        """
        time.sleep(random.uniform(start_range, end_range))

    def screenshot(self, save_path: str) -> None:
        """Takes a screenshot of the browser and saves it to a save path

        Args:
            save_path (str): path to where to save the screenshot
        """
        self.driver.save_screenshot(save_path)

    def close_browser(self) -> None:
        """Closes the browser."""
        self.driver.close()

    def add_text_input(self, by: str, value: str, text: str, wait: bool = True) -> None:
        """Adds a text input to a web element.

        Args:
            by (str): Identifier of the element type
            value (str): value of the identifier
            text (str): text to be inserted into the text input element
            wait (bool, optional): defines if one should wait for element to be found. Defaults to True.
        """
        if wait:
            field = self._wait_until_present(by, value)
            if field is None:
                return
        else:
            field = self.driver.find_element(by=by, value=value)
        field.send_keys(text)

    def click_button(self, by: str, value: str, wait: bool = True) -> None:
        """Clicks a button on the page.

        Args:
            by (str): Identifier of the element type
            value (str): value of the identifier
            wait (bool, optional): defines if one should wait until element can be clicked. Defaults to True.
        """
        if wait:
            button = self._wait_until_click(by, value)
            if button is None:
                return
        else:
            button = self.driver.find_element(by=by, value=value)
        button.click()

    def click_execute(self, by: str, value: str, wait: bool = True) -> None:
        """Performs a clicks script on the page.

        Args:
            by (str): Identifier of the element type
            value (str): value of the identifier
            wait (bool, optional): defines if one should wait for element to be found. Defaults to True.
        """
        if wait:
            button = self._wait_until_present(by, value)
            if button is None:
                return
        else:
            button = self.driver.find_element(by=by, value=value)
        self.driver.execute_script("arguments[0].click();", button)

    def iframe_switch(self, value: str) -> None:
        """Switch to a specific iframe.

        Args:
            value (str): iframe name to switch to.
        """
        self.driver.switch_to.frame(value)

    def find_element(self, by: str, value: str) -> Any:
        """Gets the element by type and returns the corresponding to it.

        Args:
            by (str): id type
            value (str): id value

        Returns:
            WebElement: element from the page's DOM tree
        """
        el = self.driver.find_element(by, value)
        return el

    def get_children(self, by: str, value: str, wait: bool = True) -> Union[Any, None]:
        """Get all children of an element.

        Args:
            by (str): Identifier of the element type
            value (str): value of the identifier
            wait (bool, optional): defines if one should wait for element to be found. Defaults to True.

        Returns:
            Union[Any, None]: web element list or None waiting fails
        """
        if wait:
            element = self._wait_until_present(by, value)
            if element is None:
                return None
        else:
            element = self.driver.find_element(by, value=value)
        return element.find_elements(By.TAG_NAME, "div")
