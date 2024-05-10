import json

from Bio import Entrez, Medline
from quart import Quart, request, jsonify
import aiohttp
import asyncio


app = Quart(__name__)

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
    name = name.capitalize()
    species_info = {}
    base_url = 'https://api.gbif.org/v1/species'
    headers = {'accept': 'application/json'}
    search_url = f'{base_url}/search'
    search_params = {'datasetKey': 'd7dddbf4-2cf0-4f39-9b2a-bb099caae36c',
                     'isExtinct': False,
                     'q': name,
                     'limit': 1}
    image_params = {'limit': 1}
    async with aiohttp.ClientSession() as session:
        async with session.get(search_url, params=search_params, headers=headers) as resp:
            search_result = await resp.json()
            if len(search_result['results']) == 0:
                return species_info
            # may get wrong result
            elif not search_result['results'][0]['scientificName'].startswith(name):
                print('bad search result')
                return species_info
        for key in ('kingdom', 'phylum', 'order', 'family', 'genus',
                    'scientificName', 'publishedIn', 'descriptions'):
            species_info[key] = search_result['results'][0][key]
        usage_key = search_result['results'][0]['key']
        image_url = f'{base_url}/{usage_key}/media'
        async with session.get(image_url, params=image_params, headers=headers) as resp:
            search_result2 = await resp.json()
            image = search_result2['results'][0]['identifier']
            species_info['image'] = image
    return species_info


@app.get('/pubmed/search')
async def pubmed_search():
    """
    Search PubMed using the provided keyword.

    Args:
        keyword (str): The search keyword.
        retmax (int, optional): The maximum number of results to return. Defaults to 10.

    Returns:
        JSONResponse: A JSON response containing the search results.
    """
    keyword = request.args.get('keyword')
    retmax = int(request.args.get('retmax'))
    records = await asyncio.to_thread(search_pubmed, keyword, retmax)
    records = search_pubmed(keyword, retmax)
    return jsonify(records)


@app.get('/gbif/search')
async def gbif_search() -> dict:
    species = request.args.get('species')
    record = await search_gbif(species)
    return jsonify(record)

