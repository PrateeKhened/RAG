#!/usr/bin/env python3

import argparse
import json
from lib.keyword_search import search 
from lib.inverted_index import build_command, tf_command, idf_command, tfidf_command 

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="get the term frequency")
    tf_parser.add_argument("doc_id", type=int, help="document id to serch the term")
    tf_parser.add_argument("term", type=str, help="term to search in the document for give doc_id")

    idf_parser = subparsers.add_parser("idf", help="get the inverse document frequency")
    idf_parser.add_argument("term", help="term to search for the idf")

    tfidf_parser = subparsers.add_parser("tfidf", help="get the term frequency-inverse document frequency")
    tfidf_parser.add_argument("doc_id", type=int, help="document id to serch the term")
    tfidf_parser.add_argument("term", type=str, help="term to search in the document for give doc_id")


    args = parser.parse_args()

    match args.command:
        case "build": 
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tfidf":
            tfidf_val = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf_val:.2f}")
        case "tf":
            tf_val = tf_command(args.doc_id, args.term)
            print(tf_val)
        case "idf":
            idf_val = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_val:.2f}")
        case "search":
            print(f"Searching for: {args.query}")
            results = search(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()