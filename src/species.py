import json
import functools
import io

from Bio import Entrez, Medline
from quart import Quart, request, jsonify, send_file, abort
import aiohttp
import asyncio


app = Quart(__name__)
with open('key', 'r') as _:
    WIKI_KEY = _.read().strip()


def search_pubmed(keyword: str, retmax: int = 5) -> list:
    """
    Search PubMed using the NCBI Entrez API.

    Args:
        keyword (str): The search keyword.
        retmax (int, optional): The maximum number of results to return. Defaults to 10.

    Returns:
        list: A list of JSONified records.
    """
    # Set API key and email (required for NCBI Entrez API)
    # Entrez.api_key = "YOUR_API_KEY"
    Entrez.email = "wpwupingwp@outlook.com"

    # Set the search query
    handle = Entrez.esearch(db="pubmed", term=keyword, retmax=retmax,
                            retmode='json')
    _ = json.loads(handle.read())
    id_list = _['esearchresult']['idlist']

    # Get the search results
    record_ids = Entrez.efetch(db="pubmed", id=','.join(id_list),
                               rettype="medline", retmode="text")

    # Parse the Medline records
    records = Medline.parse(record_ids)

    # Convert the records to JSON
    json_records = []
    for record in records:
        json_record = {
            "pmid": record.get("PMID", ""),
            "title": record.get("TI", ""),
            "journal": record.get("SO", ""),
            "year": record.get("DP", ""),
            "abstract": record.get("AB", "")
        }
        json_records.append(json_record)

    print(json_records)
    return json_records


async def search_gbif(name: str) -> dict:
    name = name.capitalize().replace('+', ' ')
    species_info = {}
    key_list =  ('kingdom', 'phylum', 'order', 'family', 'genus',
                 'scientificName', 'publishedIn', 'descriptions', 'image')
    for key in key_list:
        species_info[key] = ''
    base_url = 'https://api.gbif.org/v1/species'
    headers = {'accept': 'application/json'}
    search_url = f'{base_url}/search'
    search_params = {'datasetKey': 'd7dddbf4-2cf0-4f39-9b2a-bb099caae36c',
                     'isExtinct': 'false',
                     'q': name,
                     'limit': 1}
    image_params = {'limit': 1}
    async with aiohttp.ClientSession() as session:
        async with session.get(search_url, params=search_params, headers=headers) as resp:
            search_result = await resp.json()
            if len(search_result['results']) == 0:
                return species_info
            # may get wrong result
            result = search_result['results'][0]
            if not result['scientificName'].startswith(name):
                print('bad search result')
                return species_info
            for key in key_list:
                species_info[key] = result.get(key, '')
            if 'descriptions' in result and len(result['descriptions']) != 0:
                species_info['descriptions'] = result['descriptions'][0]['description']
        usage_key = search_result['results'][0]['key']
        image_url = f'{base_url}/{usage_key}/media'
        async with session.get(image_url, params=image_params, headers=headers) as resp:
            search_result2 = await resp.json()
            if len(search_result2['results']) != 0:
                image = search_result2['results'][0]['identifier']
                species_info['image'] = image
    return species_info


@app.get('/pubmed/search/<string:keyword>')
async def pubmed_search(keyword: str, retmax=5):
    """
    Search PubMed using the provided keyword.

    Args:
        keyword (str): The search keyword.
        retmax (int, optional): The maximum number of results to return. Defaults to 5

    Returns:
        JSONResponse: A JSON response containing the search results.
    """
    retmax = max(int(request.args.get('retmax')), retmax)
    records = await asyncio.to_thread(search_pubmed, keyword, retmax)
    return jsonify(records)


@app.get('/gbif/search/<string:species>')
async def gbif_search(species) -> dict:
    record = await search_gbif(species)
    return jsonify(record)


@app.get('/gbif/map/<string:tax_id>')
async def gbif_map(tax_id: str):
    base_url = 'https://api.gbif.org/v2/map/occurrence/density'
    # fixed options
    parameters = (f'/0/0/0%401x.png?srs=EPSG%3A4326'
                  '&style=greenHeat.point&taxonKey={tax_id}')
    return await app.redirect(base_url+parameters)


@functools.lru_cache(maxsize=1024)
@app.get('/wiki/image/<string:name>')
async def wiki_image(name: str):
    base_url = 'https://api.wikimedia.org/core/v1'
    query_url = f'https://en.wikipedia.org/w/api.php'
    parameters = dict(titles=name, action='query', format='json', prop='images')
    headers = {'Authorization': f'Bearer {WIKI_KEY}', 'User-Agent': 'test (wp)'}
    async with aiohttp.ClientSession() as session:
        async with session.get(query_url, headers=headers, params=parameters) as resp:
            search_result = await resp.json()
            if len(search_result['query']['pages']) != 0:
                record = search_result['query']['pages'].popitem()[1]
                if 'images' not in record:
                    return abort(404, 'No image found.')
                img_title = record['images'][0]['title']
                img_info_url = f'{base_url}/commons/file/{img_title}'
            else:
                return abort(404, 'No image found.')
        async with session.get(img_info_url, headers=headers) as resp:
            img_info = await resp.json()
            img_url = img_info['preferred']['url']
        async with session.get(img_url, headers=headers) as resp:
            img = await resp.content.read()
            img_type = resp.content_type
            img_file = io.BytesIO(img)
            img_file.seek(0)
    return await send_file(img_file, mimetype=img_type)


@functools.lru_cache(maxsize=1024)
@app.get('/wiki/page/<string:name>')
async def wiki_page(name: str):
    base_url = 'https://api.wikimedia.org/core/v1'
    headers = {'Authorization': f'Bearer {WIKI_KEY}', 
               'User-Agent': 'test (wp)'}
    en_url = f'{base_url}/wikipedia/en/search/title'
    parameters = {'q': name, 'limit': 1}
    # parameters2 = {'content_model': 'plaintext'}
    parameters2 = {'prop': 'extracts', 'exlimit': 'max', 'explaintext': 'true',
                   'action': 'query', 'format': 'json'}
    async with aiohttp.ClientSession() as session:
        async with session.get(en_url, headers=headers, params=parameters) as resp:
            search_result = await resp.json()
            if len(search_result.get('pages', [])) != 0:
                page_title = search_result['pages'][0]['title']
                page_id = search_result['pages'][0]['id']
                parameters2['titles'] = page_title
                # new api do not support content model
                page_url = f'https://en.wikipedia.org/w/api.php'
            else:
                return ''
        async with session.get(page_url, headers=headers, params=parameters2) as resp2:
            page = await resp2.json()
            # page_source = page['source']
            page_text = page['query']['pages'][str(page_id)]['extract']
            return page_text



@app.get('/species/info')
async def species_info() -> dict:
    llm_url = 'http://1.14.109.84:8188/v1'
    pass
