from pathlib import Path


with open('key', 'r') as _:
    username = _.readline().strip()
    password = _.readline().strip()
    member_id = _.readline().strip()
    uuid_ = _.readline().strip()
timeout = 60
output = Path(r'F:\Rosaceae_img_test')
