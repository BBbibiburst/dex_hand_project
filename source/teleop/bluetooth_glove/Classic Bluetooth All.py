import bluetooth

def parse_line(line: str):
    line = line.strip()
    if not line:
        return None, None

    tag = line[0]
    body = line[1:]

    try:
        values = [float(v) for v in body.split(":")]
    except:
        return None, None

    return tag, values


sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect(("20:20:11:11:16:22", 1))

buffer = ""

state = {
    "F": None,  # 5 dims
    "A": None,  # 3 dims
    "G": None   # 3 dims
}

while True:
    data = sock.recv(1024).decode(errors="ignore")
    buffer += data

    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        tag, values = parse_line(line)

        if tag in state:
            state[tag] = values

        # 当三种都齐了，就输出一帧 11 维
        if state["F"] and state["A"] and state["G"]:
            full_vec = state["F"] + state["A"] + state["G"]
            print("11D Frame:", full_vec)
