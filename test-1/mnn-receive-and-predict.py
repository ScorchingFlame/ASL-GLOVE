import serial, asyncio, numpy as np, json
from tensorflow.keras.models import load_model

port = input("Serial: ")
baud = 115200
modeldir = input("Model Dir: ")
ser = serial.Serial(port=port, baudrate=baud)
loop = asyncio.get_event_loop()
infojson = {}
with open(modeldir + "info.json", "r") as fp:
    infojson: dict = json.load(fp)
models = {x:load_model(f"{modeldir}{x}.h5") for x in infojson.keys()}

models_with_labels = {}
for x in infojson:
    labels = []
    for y in infojson[x]:
        with open(y, "r") as fw:
            i:dict = json.load(fw)
            _ = [labels.append(z) for z in i.keys()]
    models_with_labels[x] = labels

async def validate_and_predict(data: list):
    e = data
    e = np.array(e)
    e = e / np.max(e)
    e = np.expand_dims(e, axis=0)
    predictions = {}
    for x in models_with_labels.values():
        for y in x:
            predictions[y] = 0.0
    # predictions = {labeles:0 for labeles in (labels for labels in models_with_labels.values())}
    for model in models.items():
        prediction = model[1].predict(e)
        # print(prediction)
        i = np.argmax(prediction)
        indexofprediction = 0
        for o in prediction[0]:
            predictions[models_with_labels[model[0]][indexofprediction]] += float(o)
            indexofprediction += 1
    print(predictions)

while True:
    iniset = []
    ser.reset_input_buffer()
    while len(iniset) < 40:
        d1at = str(ser.readline())[2:][:-5].split("|")
        d1at.pop()
        prevX = []
        for x in d1at:
            l = x.split(",")
            prevX = l
            try:
                iniset.append([float(y) for y in l])
            except:
                iniset.append([float(y) for y in prevX])
    loop.run_until_complete(validate_and_predict(iniset))
    