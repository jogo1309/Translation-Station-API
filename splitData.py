from pathlib import Path

path = Path(__file__).parent / 'models' / 'test data' / 'FR' / 'europarl-v7-FR.txt'

f = open(path)
fpath = Path(path).parent
for index,line in enumerate(f.readlines()):
    newF = open("models\\test data\\FR\\FR line "+ str(index+1)+".txt", "w+")
    newF.write(line)
    newF.close
    print(line)

