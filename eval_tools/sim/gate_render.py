from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
for _ in range(10):
    app.update()
print("GATE_OK", flush=True)
app.close()
