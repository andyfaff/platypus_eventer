#!/usr/bin/env python3
import asyncio
import zmq.asyncio
import json
import argparse
import configparser
import aiofiles


async def zmq_logger(args):
    ctx = zmq.asyncio.Context()
    zmq_socket = ctx.socket(zmq.SUB)
    zmq_socket.connect(args.zmq_addr)
    zmq_socket.subscribe("")

    print(f"[*] Subscribed to ZMQ at {args.zmq_addr}")
    print(f"[*] Watching tags: {args.tags}")

    try:
        # Open file in append mode
        async with aiofiles.open(args.outfile, mode='a') as f:
            while True:
                msg = await zmq_socket.recv_json()
                name = msg.get("name")
                if name:
                    # Filter logic: Check against interesting_tags or /control/ prefix
                    if name in args.tags:  # or name.startswith("/control/"):
                        # print(msg)
                        await f.write(f"{name}, {msg['ts']}, {msg['value']}\n")
                        await f.flush()  # Ensure it's written to disk

    except asyncio.CancelledError:
        print("\n[!] Shutting down bridge...")
    except Exception as e:
        print(f"\n[!] Error: {e}")
    finally:
        zmq_socket.close()
        ctx.term()


def main():
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(description="ZMQ Logger")

    # ZMQ Connection Args
    parser.add_argument("--zmq-addr", default="tcp://<SERVER-NAME>:5566", help="ZMQ source address")

    # File and Filter Args
    parser.add_argument("--outfile", default="instrument_log.dat", help="File to save full records")
    parser.add_argument("--tags", nargs="+", required=False,
                        help="List of interesting tags to filter (space separated)")

    args = parser.parse_args()
    print(args.tags)

    try:
        asyncio.run(zmq_logger(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
