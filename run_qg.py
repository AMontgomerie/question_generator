import argparse
from questiongenerator import QuestionGenerator
from questiongenerator import print_qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answer_style",
        default="all",
        type=str,
        help="The desired type of answers. Choose from ['all', 'sentences', 'multiple_choice']",
    )
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--show_answers", dest="show_answers", action="store_true", default=True)
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--use_qa_eval", dest="use_qa_eval", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.text_file, 'r') as file:
        text_file = file.read()
    qg = QuestionGenerator()
    qa_list = qg.generate(
        text_file,
        num_questions=int(args.num_questions),
        answer_style=args.answer_style,
        use_evaluator=args.use_qa_eval
    )
    print_qa(qa_list, show_answers=args.show_answers)
