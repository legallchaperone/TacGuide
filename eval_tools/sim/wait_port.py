import sys, time, websockets.sync.client as c, msgpack
p=int(sys.argv[1])
while True:
    try:
        ws=c.connect(f"ws://127.0.0.1:{p}", open_timeout=8, max_size=None); print(p, msgpack.unpackb(ws.recv(), raw=False), flush=True); ws.close(); break
    except Exception: time.sleep(15)
