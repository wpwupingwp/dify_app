from pathlib import Path
from shutil import copy

sort_score = 'sort_output.txt'
img_folder = '/home/yaoxuan/tests/pics'
id_score = list()
with open(sort_score, 'r') as _:
    for line in _:
        x = eval(line.rstrip())
        x = x[0]
        if len(x)!=2:
            print(x)
        # 0.9,0.8,0.7
        # group, id, score
        id_score.append([str(x[1])[2], x[0], x[1]])
img_files = dict()
p = Path(img_folder)
for i in p.rglob('*.jpg'):
    img_files[i.stem] = i
last_group = '?'
n = 0
result_file = 'result.csv'
out = open(result_file, 'w')
for record in id_score:
    group, file_id, score = record
    if group != last_group:
        last_group = group
        n = 0
    if n >= 10:
        continue
    img_file = img_files[file_id]
    new_file = group + '-' + img_file.name
    out.write(f'{group},{score},{img_file.name}\n')
    copy(img_file, new_file)
    n += 1


