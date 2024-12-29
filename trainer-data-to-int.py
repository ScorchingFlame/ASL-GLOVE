import json

data = {}
fname = input("file name: ")
with open(fname, "r") as file:
    data = json.load(file)
    file.close()

newData = {}
for x in data:
    newData[x] = [[[float(a) for a in z] for z in y] for y in data[x]]

with open(fname, "w") as file2:
    json.dump(newData, file2)
    file2.close()