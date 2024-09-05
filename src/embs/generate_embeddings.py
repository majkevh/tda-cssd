from elmo import ContextEmbedderELMo
from bert import ContextEmbedderBERT
from xlmr import ContextEmbedderXLMR
import argparse



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Topological summary for embeddings.")
    parser.add_argument("--language", "-l", required=True, choices=["english", "german", "latin", "swedish"], help="Language choice")
    parser.add_argument("--embedding", "-e", required=True, choices=["BERT", "ELMo", "XLM-R"], help="Embedding choice")
    parser.add_argument("--corpus", "-c", required=True, choices=["corpus1", "corpus2"], help="Corpus selection")
    args = parser.parse_args()

    if args.embedding == "ELMo":
        processor = ContextEmbedderELMo(language=args.language, corpus=args.corpus)
        processor.run()
    elif args.embedding == "BERT":
        processor = ContextEmbedderBERT(language=args.language, corpus=args.corpus)
        processor.run()
    elif args.embedding == "XLM-R":
        processor = ContextEmbedderXLMR(language=args.language, corpus=args.corpus)
        processor.run()
    else:
        raise NotImplementedError(f"{args.embedding} is not supported.")

if __name__ == "__main__":
    main()