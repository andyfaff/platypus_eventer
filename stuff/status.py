import urllib.request


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
    def __init__(self, auth_user, auth_passwd="", url=""):
        self.passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        self.passman.add_password(None, self.url, auth_user, auth_passwd)
        self.authhandler = urllib.request.HTTPBasicAuthHandler(self.passman)
        self.opener = urllib.request.build_opener(self.authhandler)
        self.url = url
        urllib.request.install_opener(self.opener)

    def __call__(self):
        with urllib.request.urlopen(self.url, timeout=5) as response:
            txt = response.read().decode("UTF-8")
        return txt
