import json
from pathlib import Path
from time import sleep
import csv
import re

from loguru import logger as log
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
# from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from config import username, password, member_id, uuid_, timeout, output

if not output.exists():
    output.mkdir()
options = Options()
# options.set_preference('profile', profile_path)
options.enable_downloads = True
options.set_preference("browser.download.folderList", 2)
options.set_preference("browser.download.manager.showWhenStarting", False)
options.set_preference("browser.download.dir", str(output))
driver = webdriver.Firefox(options=options)
# options = webdriver.ChromeOptions()
# options.add_argument("--start-maximized")
# options.add_argument("--headless=new")
# prefs = {'download.default_directory': str(output), 'directory_upgrade': True,
#          'download.prompt_for_download': False,
#          'download.directory_upgrade': True, 'safebrowsing.enabled': True}
# options.add_experimental_option('prefs', prefs)
# options.enable_downloads = True
# driver = webdriver.Chrome(options=options)
action = webdriver.ActionChains(driver)
# driver = webdriver.Firefox()

# 300 img per species
max_n_img = 600
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
    cookies = driver.get_cookies()
    c = driver.get_cookies()
    log.info(repr(c))
    return cookies


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


def get_img_links(species_url: str) -> tuple:
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
    return tuple(img_links)


def get_name_links(species_urls: list, skip=False) -> Path:
    json_file = Path('name_links.json')
    if skip:
        return json_file
    name_links = dict()
    for species_url in species_urls:
        url = species_url['url']
        name = species_url['latin']
        img_links = get_img_links(url)
        name_links[name] = img_links
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(name_links, f, indent=True)
    return json_file


def get_img(img_link: str) -> Path:
    driver.get(img_link)
    img_logo = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, 'viewer2')))
    action.move_to_element(img_logo).perform()
    img_btn = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((By.CLASS_NAME, 'img_yuantu')))
    before = set(output.glob('*.jpg'))
    img_btn.click()
    while True:
        after = set(output.glob('*.jpg'))
        increment = after - before
        if increment:
            log.info(f'Got {increment} of {len(increment)}')
            break
        else:
            sleep(0.5)
    # img = img_btn.get_attribute('href')
    # driver.download_file(img, str(output))
    return increment.pop()


def main():
    log.info('Start')
    # account limit: 1000 img/day
    login2()
    # login()
    species_list = get_species_list()
    species_urls = get_species_urls(species_list, skip=True)
    name_links_file = get_name_links(species_urls, skip=True)
    history = Path('url_file.json')
    history2 = Path('url_file_test_set.json')
    if history.exists():
        url_file = json.loads(history.read_text())
        log.info(f'Load {len(url_file)} records')
        if history2.exists():
            url_file.update(json.loads(history2.read_text()))
    else:
        url_file = dict()
    with open(name_links_file, 'r', encoding='utf-8') as f:
        name_links = json.load(f)
    for name in name_links:
        log.info(name)
        skip = 0
        for link in name_links[name]:
            if link in url_file:
                skip += 1
                log.info(f'Skip {skip}: {link}')
                continue
            log.info(f'Download from {link}')
            try:
                filename = get_img(link)
                url_file[link] = str(filename)
            except Exception:
                log.critical(f'{link} failed')
            finally:
                history2.write_text(json.dumps(url_file))
    driver.quit()
    log.info(f'{output=}')
    log.info('Done')


if __name__ == '__main__':
    main()