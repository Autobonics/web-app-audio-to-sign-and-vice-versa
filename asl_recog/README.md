# asl-recog
LSTM model to recognize ASL gloss

### Data collection
- create venv using 

```powershell
python -m venv asl-recog-venv
```

- Activate virtual environment
    - windows powershell
        ```powershell
        .\asl-recog-venv\Scripts\Activate.ps1
        ```
    - Linux bash
        ```bash
        source ./asl-recog-venv/bin/activate
        ```
- install the requirements using pip

```bash
    pip install -r requirements.txt
```
- Run the cli data generation tool

```bash
python gen_cli.py  -h 
```

- Add the list of gloss in gloss.txt file and invoke the cli program with -r argument

```bash
python gen_cli.py  -r
```

OR

- provide each gloss name as argument with -g flag

```bash
python gen_cli.py  -g "Happy Birthday"
```

- Also use -v flag to change the number of videos recorded for each gloss

```bash
python gen_cli.py  -v 15
```
- Also use -f flag to change the number of frames in each video

```bash
python gen_cli.py  -f 30
```

