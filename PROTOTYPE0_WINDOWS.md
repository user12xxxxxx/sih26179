# Prototype 0 on the Lenovo (Windows)

One laptop does everything. It prepares the AI model, runs the in-store "brain" and shows the
dashboard. Your phone is the camera. It takes about one evening and costs nothing.

## 1. One-time setup (about 20 minutes)

1. Install **Python 3.12** from python.org. On the first screen of the installer, tick
   **"Add python.exe to PATH"**.
2. Unzip `drishti.zip` to a short path, for example `C:\sih\drishti`.
3. Open that folder in File Explorer, click the address bar, type `powershell` and press Enter.
4. Run these lines one at a time:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install ultralytics onnxruntime
   pip install -e ".[dev]"
   ```

   If the second line says "running scripts is disabled", run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, answer `Y`, and try again.
   Every time you open a new PowerShell window later, run `.venv\Scripts\Activate.ps1` first.

5. Get the free people-spotting model (it downloads once, then converts it):

   ```powershell
   yolo export model=yolov8n.pt format=onnx
   mkdir models
   move yolov8n.onnx models\
   ```

## 2. Check everything works (5 minutes)

```powershell
python -m drishti run configs/synthetic_demo.yaml
python -m drishti serve --db out/synthetic/events.db
```

Open http://127.0.0.1:8000 to see the simulated store. Press Ctrl+C in PowerShell to stop the
dashboard. If Windows asks whether Python may use the network, "Private networks" is enough.

## 3. Film two videos (15 minutes)

- Set the phone camera to **1080p, not 4K**. Prop it up high and keep it still, looking down.
- **Video A:** the doorway, 3 to 5 minutes, with friends walking in and out. Count entries and exits by hand.
- **Video B:** a table standing in for the billing counter. Friends queue behind it and take turns
  "paying" for about 30 seconds each. Note roughly how long each person waited.
- Make a `clips` folder in `C:\sih\drishti` and copy the videos there as `clips\door.mp4` and `clips\counter.mp4`.

## 4. Show it where the door and the counter are (20 minutes)

```powershell
python -m drishti snapshot clips/door.mp4
python -m drishti snapshot clips/counter.mp4
```

Open `clips\door_grid.jpg` and `clips\counter_grid.jpg`. The yellow numbers are pixel positions:
x across, y down. Open `configs\prototype0.yaml` in Notepad and replace the example numbers:

- `door`: the two ends of a line across the doorway, on the floor, then any point inside the room.
- `queue_1`: four corners of the area where people stand waiting.
- `counter_1`: four corners of the area where the person paying stands.

## 5. Run prototype 0 (10 minutes)

```powershell
python -m drishti run configs/prototype0.yaml
python -m drishti report --db out/p0/events.db
python -m drishti serve --db out/p0/events.db
```

The terminal prints each entry, exit and finished "payment" as it happens. The report and the
dashboard show the totals. Compare "Visitors in" with your hand count: that is your first result.

## If the numbers are off

| What you see | Try |
| --- | --- |
| People missed | Lower `conf` from 0.4 to 0.3 on that camera's detector in `prototype0.yaml` |
| Someone counted twice | Move the door line to where people walk straight across it; mount the phone higher |
| Entries counted as exits | `in_point` is on the wrong side of the line |
| Queue always 0 | The `queue_1` corners don't cover where people's feet are; the software uses the feet position |
| "cannot open clips/door.mp4" | Check the file name, and re-record at 1080p if the phone saved HEVC/4K |

Later, and optional: the RTX 4050 can run the model (`pip install onnxruntime-gpu`, then add
`providers: [CUDAExecutionProvider]` to each detector). Prototype 0 doesn't need it; the i7 is plenty.
