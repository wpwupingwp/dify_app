import json
from pathlib import Path
from time import sleep
import csv
import re

from loguru import logger as log
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import username, password, member_id, uuid_

timeout = 60
output = Path(r'F:\Rosaceae_img')
if not output.exists():
    output.mkdir()
options = Options()
# options.set_preference('profile', profile_path)
options.enable_downloads = True
options.set_preference("browser.download.folderList", 2)
options.set_preference("browser.download.manager.showWhenStarting", False)
options.set_preference("browser.download.dir", str(output))
driver = webdriver.Firefox(options=options)
action = webdriver.ActionChains(driver)
# driver = webdriver.Firefox()

# 300 img per species
max_n_img = 300
search_url = 'https://ppbc.iplant.cn/search'


def login():
    driver.get(search_url)
    driver.add_cookie({'name': 'MemberId', 'value': member_id})
    driver.add_cookie({'name': 'uuid', 'value': uuid_})
    c = driver.get_cookies()
    log.info(repr(c))
    return


def login2():
    login_id = 'loginbtn'
    submit_button_id = 'btnlog'
    name_id = 'userInput'
    pwd_id = 'userPwd'
    user_id = 'loginname'
    driver.get(search_url)
    login_field = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, login_id)))
    login_field.click()
    name_field = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, name_id)))
    pwd_field = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, pwd_id)))
    submit_button = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, submit_button_id)))
    name_field.send_keys(username)
    pwd_field.send_keys(password)
    submit_button.click()
    user_id = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, user_id)))
    log.info(f'Logged as {user_id}')
    cookie = driver.get_cookies()
    return cookie


def get_species_list(list_file='species_list.csv') -> list:
    # [Scientific name, Chinese name]
    with open(list_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)
        return data


def get_species_urls(species_list: list, skip=False):
    json_file = 'species_url.json'
    if skip:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    species_urls = list()
    for record in species_list:
        latin, cn = record[0], record[1]
        url = search_name(latin)
        species_urls.append({'latin': latin, 'cn': cn, 'url': url})
        if skip:
            break
    with open(json_file, 'w') as _:
        json.dump(species_urls, _, indent=True)
    return species_urls


def search_name(name: str):
    search_id = 'txtlatin'
    submit_button_id = "btnok"
    driver.get(search_url)
    name_field = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, search_id)))
    submit_button = driver.find_element(By.ID, submit_button_id)
    name_field.send_keys(name)
    submit_button.click()
    WebDriverWait(driver, timeout).until(EC.url_changes(search_url))
    new_url = driver.current_url
    log.info(f'{name} {new_url}')
    return new_url


def get_img_links(species_url: str) -> set:
    # Scroll to the bottom of the page until all content is loaded
    img_links = set()
    driver.get(species_url)
    while True:
        # species count load after few seconds
        spcount_ = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.ID, 'spcount')))
        spcount_raw = re.search(r'\d+', spcount_.text)
        if spcount_raw:
            spcount = int(spcount_raw.group(0))
            break
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # 5 seconds enough?
        WebDriverWait(driver, 5.0)
        links_ = driver.find_elements(By.TAG_NAME, 'a')
        hrefs = [i.get_attribute('href') for i in links_]
        log.info(f'{hrefs=}')
        links = [i for i in hrefs if '/tu/' in i]
        img_links |= set(links)
        if len(img_links) >= max_n_img or len(img_links) >= spcount:
            log.info(f'Got enough ({len(img_links)}) links from {species_url}')
            break
        # else:
        #     last_height = new_height
    return img_links


def get_img(img_link: str):
    driver.get(img_link)
    img_logo = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, 'viewer2')))
    action.move_to_element(img_logo).perform()
    img_btn = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((By.CLASS_NAME, 'img_yuantu')))
    img_btn.click()
    img = img_btn.get_attribute('href')
    log.info(f'Downloading {img}')
    return


def main():
    login2()
    log.info('Start')
    species_list = get_species_list()
    species_urls = get_species_urls(species_list, skip=True)
    for species_url in species_urls:
        url = species_url['url']
        img_links = get_img_links(url)
        for link in img_links:
            get_img(link)
            sleep(0.5)
        break
    driver.quit()
    log.info(f'{output=}')
    log.info('Done')


if __name__ == '__main__':
    main()