from pathlib import Path

for i in Path().glob('*'):
    if i.is_dir():
        try:
            i.rmdir()
        except:
            pass

