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


class Status:
    def __init__(self, user, password="", url=""):
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

    def from_file(self, pth, dataset=0):
        pth = Path(pth)
        with open(pth / f"DATASET_{dataset}" / "final_state.txt", "r") as f:
            txt = "".join(f.readlines())
        return txt
