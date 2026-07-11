import bluetooth
import time

def parse_f_line(line: str):
    line = line.strip()
    if not line.startswith("F"):
        return None

    body = line[1:]
    try:
        values = [int(v) for v in body.split(":")]
        if len(values) == 5:
            return values
    except:
        pass
    return None

# ========== 蓝牙连接 ==========
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect(("20:20:11:11:16:22", 1))
print("Connected to glove")

buffer = ""

# ========== 标定数据 ==========
min_vals = None
max_vals = None

def collect_samples(prompt, duration=3.0):
    print("\n" + prompt)
    print(f"请保持姿态 {duration} 秒...")
    samples = []
    start = time.time()
    while time.time() - start < duration:
        data = sock.recv(1024).decode(errors="ignore")
        global buffer
        buffer += data
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            vals = parse_f_line(line)
            if vals:
                samples.append(vals)
    return samples


# ========== 标定流程 ==========

# 2️⃣ 握拳
fist_samples = collect_samples("👉 请【用力握紧拳头】", 3.0)
max_vals = [max(col) for col in zip(*fist_samples)]
print("Max (Fist):", max_vals)

# 1️⃣ 张开手
open_samples = collect_samples("👉 请【完全张开手掌】", 3.0)
min_vals = [min(col) for col in zip(*open_samples)]
print("Min (Open hand):", min_vals)



print("\n✅ 标定完成！\n")

# ========== 实时归一化 ==========
def normalize(vals, vmin, vmax):
    norm = []
    for x, mn, mx in zip(vals, vmin, vmax):
        if mx > mn:
            y = (x - mn) / (mx - mn)
        else:
            y = 0.0
        y = max(0.0, min(1.0, y))
        norm.append(y)
    return norm


print("开始输出归一化后的 5 维弯曲程度 (0~1)：\n")

while True:
    data = sock.recv(1024).decode(errors="ignore")
    buffer += data

    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        vals = parse_f_line(line)
        if vals:
            norm_vals = normalize(vals, min_vals, max_vals)
            print("Raw:", vals, " -> Norm:", ["%.2f" % v for v in norm_vals])
