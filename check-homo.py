import json

data = {}
fname = input("File: ")
with open(fname, "r") as file:
    data = json.load(file)
    file.close()
a = 25
b = 40
c = 12
ai = 0
bi = 0
ci = 0
for x in data:
    if len(data[x]) == a:
        for y in data[x]:
            if len(y) == b:
                for z in y:
                    if len(z) == c:
                        pass
                    else:
                        print("no1", ci)
                    ci += 1
                ci = 0
            else:
                print(x)
                print(len(y))
                print("no2", bi)
            bi += 1
        bi = 0
    else:
        print("no3", ai)
    ai += 1