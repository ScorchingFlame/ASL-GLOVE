import serial, json

port = input("Serial: ")
baud = int(input("Baud rate: "))
label = input("Label: ")
filename = f"./training-data-{label}.json"
ser = serial.Serial(port=port, baudrate=baud)
dat = {
    label: []
}
i = 1
while True:
    e = input("Enter to continue or enter q to quit or d to discard last one: ")
    if  e == "q":
        break
    elif e == "d":
        dat[label].pop()
        i -= 1
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
    dat[label].append(iniset)
    print(f"Collected {i} amount of data.")
    i += 1

with open(filename, "w") as file:
    json.dump(dat, file)
    