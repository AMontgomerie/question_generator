import argparse
import numpy as np
from questiongenerator import QuestionGenerator

def print_qa(qa_list):
    if len(qa_list) == 0:
        print("No questions generated.")
    for i in range(len(qa_list)):
        print('{}) Q: {}'.format(i + 1, qa_list[i]['question']))
        print('{}A:'.format(' ' * int(np.where(i < 9, 3, 4))), qa_list[i]['answer'], '\n')

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
        "--num_questions",
        default=10,
        type=int,
        help="The desired number of questions to generate.",
    )
    args = parser.parse_args()

    with open(args.text_dir, 'r') as file:
        text_file = file.read()

    qg = QuestionGenerator()

    print("\nGenerating questions...")
    qa_list = qg.generate(text_file, args.num_questions)
    print_qa(qa_list)

if __name__ == "__main__":
    main()