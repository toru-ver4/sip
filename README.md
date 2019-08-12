# sip

Sample codes for SIgnal Processing.

## Installation (w/o OpenImageIO)

### Command

```bash
git clone https://github.com/toru-ver4/sip.git
cd sip
python -m venv venv
source ./venv/bin/activate
pip install --upgrade pip && wget -O - https://github.com/toru-ver4/docker/raw/develop/python_environment/requirements.txt | pip install -r /dev/stdin
```

### Remarks

* This project recommends to use **venv** virtual environment.
* And I recommend to add ```source /work/bin/activate``` statement to the ```.bashrc```.

## Installation (w/ OpenImageIO)

see https://github.com/toru-ver4/docker/tree/develop/python_environment .
