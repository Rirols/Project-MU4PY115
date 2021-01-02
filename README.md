# Molecular dynamics project

## Description

Neural network that takes particles positions as input and gives system energy as output

## Installation

Python virtualenv creation :

```
$ mkdir md_env
$ python3 -m venv md_env/
$ source md_env/bin/activate
```

Python packages installation :

```
(md_env) $ pip install --upgrade pip setuptools wheel
(md_env) $ pip install numpy matplotlib tensorflow scikit-learn keras ase dscribe
```

## Usage

Simply run the `main.py`script :

```
(md_env) $ python3 main.py
```

Alternatively :

```
(md_env) $ chmod +x main.py
(md_env) $ ./main.py
```

## Authors

* **Lorris Giovagnoli**
* **Pierre Houzelstein**
