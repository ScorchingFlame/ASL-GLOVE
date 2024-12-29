import json
e = {}
with open(".\\test1.json", "r") as fw:
    e = json.load(fw)

c = e

for x in e: #labels
    i=0
    for y in range(25): # 25
        for z in range(40):
            c[x][y][z].pop()

with open(".\\test.json", "w") as fwah:
    json.dump(c, fwah)