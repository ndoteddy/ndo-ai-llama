"""Streamlit tool to inspect a persisted Chroma DB's collections.

Run with: streamlit run scripts/viewer.py -- <db_path>
"""
import argparse
import logging
import sys

import chromadb
import pandas as pd
import streamlit as st

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

parser = argparse.ArgumentParser(description="View the collections in a Chroma DB")
parser.add_argument("db", nargs="?", default="./data/chromadb", help="Path to the persisted Chroma DB directory")

pd.set_option("display.max_columns", 4)


def view_collections(db_path: str) -> None:
    st.markdown("### DB Path: %s" % db_path)

    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()
    st.header("Collections")

    for collection in collections:
        data = collection.get()
        df = pd.DataFrame.from_dict(data)
        st.markdown("### Collection: **%s**" % collection.name)
        st.dataframe(df)


if __name__ == "__main__":
    args = parser.parse_args()
    logging.info("opening database: %s", args.db)
    view_collections(args.db)
