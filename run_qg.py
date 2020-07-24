import argparse
import numpy as np
from questiongenerator import QuestionGenerator
from questiongenerator import print_qa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_dir",
        default=None,
        type=str,
        required=True,
        help="The text that will be used as context for question generation.",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        help="The folder that the trained model checkpoints are in.",
    )
    parser.add_argument(
        "--num_questions",
        default=10,
        type=int,
        help="The desired number of questions to generate.",
    )
    parser.add_argument(
        "--answer_style",
        default="all",
        type=str,
        help="The desired type of answers. Choose from ['all', 'sentences', 'multiple_choice']",
    )
    args = parser.parse_args()

    with open(args.text_dir, 'r') as file:
        text_file = file.read()

    qg = QuestionGenerator(args.model_dir)

    print("\nGenerating questions...")
    qa_list = qg.generate(text_file, args.num_questions, args.answer_style)
    print_qa(qa_list)

if __name__ == "__main__":
    main()