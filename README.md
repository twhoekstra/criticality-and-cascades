# criticality-and-cascades
Criticality and Cascades

# Install

Prerequisites:
- Git
- Python

1. Clone the repository
```shell
git clone https://github.com/twhoekstra/criticality-and-cascades.git
cd criticality-and-cascades
```

2. Create a virtual environment (optional but recommended). 

**Using `venv`** 
```shell
python -m venv .venv
```

Activate the virtual environment. 
**On Windows:**
```shell
.venv\Scripts\activate.bat
```

**For Unix:**
```shell
source .venv/bin/activate
```

4. Install the requirements
```shell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For more info, 
consult the [Python Packaging User Guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)


**Using Anaconda**
```shell
conda create --name criticality -c conda-forge --file=requirements.txt
conda activate criticality
```

5. Open Jupyter Lab:
```shell
jupyter lab
```

# Development
Please work in the `dev` branch of the repository.
