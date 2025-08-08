import urllib.request
from pathlib import Path


def parse_status(txt):
    output = dict()
    units = dict()
    for line in txt.split("\n"):
        if not line:
            # EOF
            break
        key, val = line.split(":")
        val = val.strip("\n ")
        bits = val.partition(" ")
        output[key] = _parse_val(bits[0])
        units[key] = bits[-1]

    return output, units


def _parse_val(v):
    try:
        if "." in v:
            val = float(v)
            return val
    except ValueError:
        pass

    try:
        val = int(v)
        return val
    except ValueError:
        pass

    return v


class State:
    """
    Parses the DAS response into dictionary form.
    """

    def __init__(self, response):
        status, units = parse_status(response)
        self.response = response
        self.dct = {}
        self.dct.update(status)

    @property
    def DAQ(self):
        return self.dct["DAQ"]

    @property
    def started(self):
        return self.dct["DAQ"] == "Started"

    @property
    def starting(self):
        return self.dct["DAQ"] == "Starting"

    @property
    def DAQ_dirname(self):
        return self.dct["DAQ_dirname"]

    @property
    def DATASET_number(self):
        return self.dct["DATASET_number"]

    @property
    def dataset_start_time_t(self):
        return self.dct["dataset_start_time_t"]


class Status:
    """
    Acquires the DAS textstatus page
    """
    def __init__(self, user="manager", password="", url=""):
        self.url = url
        self.passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        self.passman.add_password(None, self.url, user, password)
        self.authhandler = urllib.request.HTTPBasicAuthHandler(self.passman)
        self.opener = urllib.request.build_opener(self.authhandler)
        urllib.request.install_opener(self.opener)

    def __call__(self):
        with urllib.request.urlopen(self.url, timeout=5) as response:
            txt = response.read().decode("UTF-8")
        return txt

    def from_file(self, daq_dirname, dataset=0, pth="."):
        loc = Path(pth) / daq_dirname / f"DATASET_{dataset}"
        with open(loc / "final_state.txt", "r") as f:
            txt = "".join(f.readlines())
        return txt
