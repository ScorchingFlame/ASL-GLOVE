import json

files = []
e = {}
while True:
    fname = input("Enter file name (q for quit): ")
    if fname == "q":
        break
    files.append(fname)
for x in files:
    with open(x, "r") as fwae:
        le: dict = json.load(fwae)
        for x in le.keys():
            e[x] = le[x]
        fwae.close()
final_name = input("Final file name: ")
with open(final_name, "w") as final:
    json.dump(e, final)