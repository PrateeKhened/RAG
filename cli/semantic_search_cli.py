#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch, chunk_text
import json
from lib.search_utils import DATA_PATH, DEFAULT_CHUNK_SIZE

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify the model with information")

    embed_parser = subparsers.add_parser("embed_text", help="ambed the text")
    embed_parser.add_argument("text", type=str, help="text to embed")

    subparsers.add_parser("verify_embeddings", help="verify the movies data embedding")

    qembed_parser = subparsers.add_parser("embedquery", help="embed the user query")
    qembed_parser.add_argument("query", type=str, help="query to convert to embed")

    search_parser = subparsers.add_parser("search", help="find your movie")
    search_parser.add_argument("query", type=str, help="query to search the movie")
    search_parser.add_argument("--limit", type=int, default=5, help="Max results to show(default is 5)")

    chunk_parser = subparsers.add_parser("chunk", help="chunk the text")
    chunk_parser.add_argument("text", type=str, help="text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="size of the chunk(default chunk size = 200)")
    chunk_parser.add_argument("--overlap", type=int, help="size of the overlap")

    args = parser.parse_args()

    match args.command:
        case "chunk":
            chunk_text(args.text, args.overlap, args.chunk_size)
        case "search":
            ss = SemanticSearch() 
            with open(DATA_PATH, "r") as f:
                movies = json.load(f)
                documents = movies["movies"]
            ss.load_or_create_embeddings(documents)
            results = ss.search(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res["title"]} (score: {res["score"]})")
                print(f"{res["description"]}\n\n")
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings() 
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model() 
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()