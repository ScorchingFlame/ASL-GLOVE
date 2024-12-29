import serial, asyncio, numpy as np, json
from tensorflow.keras.models import load_model

port = input("Serial: ")
baud = int(input("Baud rate: "))
modelname = input("Model File: ")
trainf = input("Training file: ")
ser = serial.Serial(port=port, baudrate=baud)
model = load_model(modelname)
loop = asyncio.get_event_loop()
with open(trainf, "r") as f:
    classes = list(json.load(f).keys())
async def validate_and_predict(data: list):
    e = []
    prevAr = []
    for x in data:
        r = []
        try:
            for y in x:
                r.append(float(y))
            e.append(r)
            prevAr = r
        except:
            e.append(prevAr)
    e = np.array(e)
    e = e / np.max(e)
    e = np.expand_dims(e, axis=0)
    prediction = model.predict(e)
    print(prediction)
    i = np.argmax(prediction)
    print(classes[i])

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
    