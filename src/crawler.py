import json
import csv

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logoru

from config import profile_path

timeout = 10
options = Options()
options.set_preference('profile', profile_path)
# driver = webdriver.Firefox(options=options,)
driver = webdriver.Firefox()

search_url = 'https://ppbc.iplant.cn/search'
search_id = 'txtlatin'
submit_button_id = "btnok"


def get_species_list(list_file='species_list.csv') -> list:
    # [Scientific name, Chinese name]
    with open(list_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)
        print(data)
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
    return new_url


def main():
    species_list = get_species_list()
    species_urls = list()
    for record in species_list:
        latin, cn = record[0], record[1]
        url = search_name(latin)
        species_urls.append((*record, url))

    with open('species_url.json', 'w') as _:
        json.dump(species_urls, _, indent=True)

    driver.quit()
