import serial, asyncio, numpy as np, json
import joblib

port = input("Serial: ")
baud = 115200
modelname = "C:\\Users\\harishritheesh\\Documents\\Projects\\Python\\ASL-GLOVE\\test 2\\test-model.h5"
trainf = input("Training file: ")
ser = serial.Serial(port=port, baudrate=baud)
svm_model = joblib.load('.\\svm_model.pkl')
scaler = joblib.load('.\\scaler.pkl')
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
    e = np.expand_dims(e, axis=0)
    e = e.flatten()
    e = e.reshape(1, -1)
    e = scaler.transform(e)
    prediction = svm_model.predict_proba(e)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    z = {classes[x]:y for x, y in enumerate(prediction[0])}
    o = 0
    for u in sorted(z.items(), key=lambda x: x[1], reverse=True):
        print(f'Predicted Gesture: {u[0]}, Confidence: {u[1]:.2f}')
    print("=============================")


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
    