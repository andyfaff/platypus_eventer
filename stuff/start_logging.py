import sys
import subprocess
import argparse
from pathlib import Path
import configparser


if __name__ == "__main__":
    # print(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("user", help="User name for the DAS server")
    parser.add_argument("password", help="Password for the DAS server")
    parser.add_argument(
        "pth", help="Location for where the data is stored", default=Path.cwd()
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        required=False,
        help="List of interesting tags to filter (space separated)",
    )

    args = parser.parse_args()
    p0 = subprocess.Popen(
        ["platypus_eventer-cli", args.user, args.password, args.pth],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    config = configparser.ConfigParser()
    pth = Path(args.pth)
    config.read(str(pth / "config.ini"))
    ics = config.get("url", "ics")

    _args = ["python", "hipadaba_logger.py", "--zmq-addr", ics, "--tags"]
    _args.extend(args.tags)
    p1 = subprocess.Popen(_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout0, stderr0 = p0.communicate()
    print(stdout0.decode())
    print(stderr0.decode())

    stdout1, stderr1 = p1.communicate()
    print(stdout1.decode())
    print(stderr1.decode())
