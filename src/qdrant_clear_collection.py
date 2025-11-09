#!/usr/bin/env python3
# qdrant_clear_collection.py

from qdrant_client import QdrantClient
import argparse

def main():
    ap = argparse.ArgumentParser("Completely delete a Qdrant collection")
    ap.add_argument("--collection", required=True, help="Collection name to delete")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    client = QdrantClient(host=args.host, port=args.port, api_key=args.api_key)
    try:
        client.delete_collection(args.collection)
        print(f"[wipe] deleted collection '{args.collection}'")
    except Exception as e:
        print(f"[wipe] error: {e}")

if __name__ == "__main__":
    main()
