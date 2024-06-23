import json
from pathlib import Path
import csv
import re

from loguru import logger as log
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import username, password

timeout = 10
output = Path(r'R:\out')
if not output.exists():
    output.mkdir()
options = Options()
# options.set_preference('profile', profile_path)
options.enable_downloads = True
driver = webdriver.Firefox(options=options)
# driver = webdriver.Firefox()

# 300 img per species
max_n_img = 10
search_url = 'https://ppbc.iplant.cn/search'
search_id = 'txtlatin'
submit_button_id = "btnok"


def login() -> None:
    # directly login with credentials
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
    submit_button = driver.find_element(By.ID, submit_button_id)
    name_field.send_keys(username)
    pwd_field.send_keys(password)
    submit_button.click()
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, user_id)))
    return


def get_species_list(list_file='species_list.csv') -> list:
    # [Scientific name, Chinese name]
    with open(list_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)
        return data


def search_name(name: str):
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


def main():
    log.info('Start')
    species_list = get_species_list()
    species_urls = list()
    for record in species_list:
        latin, cn = record[0], record[1]
        url = search_name(latin)
        species_urls.append({'latin': latin, 'cn': cn, 'url': url})
        # todo,
        break

    with open('species_url.json', 'w') as _:
        json.dump(species_urls, _, indent=True)
    a = species_urls[0]['url']
    login()
    b = get_img_links(a)
    c = get_img(b.pop())
    driver.quit()
    log.info('Done')


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
        print(hrefs)
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
    img_btn = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'img_yuantu')))
    img = img_btn.get_attribute('href')
    log.info(f'Downloading {img}')
    driver.get(img)
    # driver.download_file(img, str(output))
    t = input('quit?')
    log.info(f'{output} {img}')


if __name__ == '__main__':
    main()