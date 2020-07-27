import os
import sys
import math
import numpy as np
import torch
import spacy
import re
import random
import json
import en_core_web_sm
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import BertTokenizer, BertForSequenceClassification

class QuestionGenerator():

    def __init__(self, model_dir=None):
        QG_PRETRAINED = 't5-base'
        QG_FINETUNED_MODEL = 'qg_model.pth'
        self.ANSWER_TOKEN = '<answer>'
        self.CONTEXT_TOKEN = '<context>'
        self.SEQ_LENGTH = 512

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.qg_tokenizer = T5Tokenizer.from_pretrained(QG_PRETRAINED)
        self.qg_tokenizer.add_special_tokens(
            {'additional_special_tokens': [self.ANSWER_TOKEN, self.CONTEXT_TOKEN]}
        )

        config = T5Config(decoder_start_token_id=self.qg_tokenizer.pad_token_id)
        self.qg_model = T5ForConditionalGeneration(config).from_pretrained(QG_PRETRAINED)
        self.qg_model.resize_token_embeddings(len(self.qg_tokenizer))

        if model_dir:
            checkpoint = torch.load(os.path.join(model_dir, QG_FINETUNED_MODEL))
        else:
            checkpoint = torch.load(os.path.join(sys.path[0], 'models/', QG_FINETUNED_MODEL))

        self.qg_model.load_state_dict(checkpoint['model_state_dict'])
        self.qg_model.to(self.device)

        self.qa_evaluator = QAEvaluator(model_dir)

    def generate(self, article, use_evaluator=True, num_questions=None, answer_style='all'):

        print("Generating questions...\n")

        qg_inputs, qg_answers = self.generate_qg_inputs(article, answer_style)
        generated_questions = self.generate_questions_from_inputs(qg_inputs)

        message = "{} questions doesn't match {} answers".format(
            len(generated_questions),
            len(qg_answers))
        assert len(generated_questions) == len(qg_answers), message

        if use_evaluator:

            print("Evaluating QA pairs...\n")

            encoded_qa_pairs = self.qa_evaluator.encode_qa_pairs(generated_questions, qg_answers)
            scores = self.qa_evaluator.get_scores(encoded_qa_pairs)
            if num_questions:
                qa_list = self._get_ranked_qa_pairs(generated_questions, qg_answers, scores, num_questions)
            else:
                qa_list = self._get_ranked_qa_pairs(generated_questions, qg_answers, scores)

        else:
            print("Skipping evaluation step.\n")
            qa_list = self._get_all_qa_pairs(generated_questions, qg_answers)

        return qa_list

    def generate_qg_inputs(self, text, answer_style):

        VALID_ANSWER_STYLES = ['all', 'sentences', 'multiple_choice']

        sentences = self._split_text(text)
        inputs = []
        answers = []

        if answer_style not in VALID_ANSWER_STYLES:
            raise ValueError(
                "Invalid answer style {}. Please choose from {}".format(
                    answer_style,
                    VALID_ANSWER_STYLES
                )
            )

        if answer_style == 'sentences' or answer_style == 'all':
            prepped_inputs, prepped_answers = self._prepare_qg_inputs(sentences, text)
            inputs.extend(prepped_inputs)
            answers.extend(prepped_answers)

        if answer_style == 'multiple_choice' or answer_style == 'all':
            prepped_inputs, prepped_answers = self._prepare_qg_inputs_MC(sentences)
            inputs.extend(prepped_inputs)
            answers.extend(prepped_answers)

        return inputs, answers

    def generate_questions_from_inputs(self, qg_inputs):
        generated_questions = []

        for qg_input in qg_inputs:
            question = self._generate_question(qg_input)
            generated_questions.append(question)

        return generated_questions

    def _split_text(self, text):
        MAX_SENTENCE_LEN = 128

        sentences = re.split('\(.|\?|\n|!\)', text)
        sentences = [s for s in sentences if len(s) > 0]

        cut_sentences = []
        for sentence in sentences:
            if len(sentence) > MAX_SENTENCE_LEN:
                sentences = sentences + re.split('[;:)]', sentence)
                cut_sentences.append(sentence)
        sentences = [s for s in sentences if s not in cut_sentences]
        return [s.strip(" ") for s in sentences]

    def _prepare_qg_inputs(self, sentences, text):
        inputs = []
        answers = []

        for sentence in sentences:
            qg_input = '{} {} {} {}'.format(
                self.ANSWER_TOKEN,
                sentence,
                self.CONTEXT_TOKEN,
                text
            )
            inputs.append(qg_input)
            answers.append(sentence)

        return inputs, answers

    def _prepare_qg_inputs_MC(self, sentences):

        spacy_nlp = en_core_web_sm.load()
        docs = list(spacy_nlp.pipe(sentences, disable=['parser']))
        inputs_from_text = []
        answers_from_text = []

        for i in range(len(sentences)):
            entities = docs[i].ents
            if entities:
                for entity in entities:
                    qg_input = '{} {} {} {}'.format(
                        self.ANSWER_TOKEN,
                        entity,
                        self.CONTEXT_TOKEN,
                        sentences[i]
                    )
                    answers = self._get_MC_answers(entity, docs)
                    inputs_from_text.append(qg_input)
                    answers_from_text.append(answers)

        return inputs_from_text, answers_from_text

    def _get_MC_answers(self, correct_answer, docs):

        entities = []
        for doc in docs:
            entities.extend([{'text': e.text, 'label_': e.label_} for e in doc.ents])

        # remove duplicate elements
        entities_json = [json.dumps(kv) for kv in entities]
        pool = set(entities_json)
        num_choices = min(4, len(pool)) - 1  # -1 because we already have the correct answer

        # add the correct answer
        final_choices = []
        correct_label = correct_answer.label_
        final_choices.append({'answer': correct_answer.text, 'correct': True})
        pool.remove(json.dumps({'text': correct_answer.text, 'label_': correct_answer.label_}))

        # find answers with the same NER label
        matches = [e for e in pool if correct_label in e]

        # if we don't have enough then add some other random answers
        if len(matches) < num_choices:
            choices = matches
            pool = pool.difference(set(choices))
            choices.extend(random.sample(pool, num_choices - len(choices)))
        else:
            choices = random.sample(matches, num_choices)

        choices = [json.loads(s) for s in choices]
        for choice in choices:
            final_choices.append({'answer': choice['text'], 'correct': False})
        random.shuffle(final_choices)
        return final_choices

    def _generate_question(self, qg_input):
        self.qg_model.eval()
        encoded_input = self._encode_qg_input(qg_input)
        with torch.no_grad():
            output = self.qg_model.generate(input_ids=encoded_input['input_ids'])
        return self.qg_tokenizer.decode(output[0])

    def _encode_qg_input(self, qg_input):
        return self.qg_tokenizer(
            qg_input,
            pad_to_max_length=True,
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

    def _get_ranked_qa_pairs(self, generated_questions, qg_answers, scores, num_questions=10):
        if num_questions > len(scores):
            num_questions = len(scores)
            print("\nWas only able to generate {} questions. For more questions, please input a longer text.".format(num_questions))

        qa_list = []
        for i in range(num_questions):
            index = scores[i]
            qa = self._make_dict(
                generated_questions[index].split('?')[0] + '?',
                qg_answers[index])
            qa_list.append(qa)
        return qa_list

    def _get_all_qa_pairs(self, generated_questions, qg_answers):
        qa_list = []
        for i in range(len(generated_questions)):
            qa = self._make_dict(
                generated_questions[i].split('?')[0] + '?',
                qg_answers[i])
            qa_list.append(qa)
        return qa_list

    def _make_dict(self, question, answer):
        qa = {}
        qa['question'] = question
        qa['answer'] = answer
        return qa


class QAEvaluator():
    def __init__(self, model_dir=None):
        QAE_PRETRAINED = 'bert-base-cased'
        QAE_FINETUNED_MODEL = 'qa_eval_model.pth'
        self.SEQ_LENGTH = 512

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.qae_tokenizer = BertTokenizer.from_pretrained(QAE_PRETRAINED)

        self.qae_model = BertForSequenceClassification.from_pretrained(QAE_PRETRAINED)

        if model_dir:
            checkpoint = torch.load(os.path.join(model_dir, QAE_FINETUNED_MODEL))
        else:
            checkpoint = torch.load(os.path.join(sys.path[0], 'models/', QAE_FINETUNED_MODEL))

        self.qae_model.load_state_dict(state_dict=checkpoint['model_state_dict'])
        self.qae_model = self.qae_model.to(self.device)

    def encode_qa_pairs(self, questions, answers):
        encoded_pairs = []
        for i in range(len(questions)):
            encoded_qa = self._encode_qa(questions[i], answers[i])
            encoded_pairs.append(encoded_qa.to(self.device))
        return encoded_pairs

    def get_scores(self, encoded_qa_pairs):
        scores = {}
        self.qae_model.eval()
        with torch.no_grad():
            for i in range(len(encoded_qa_pairs)):
                scores[i] = self._evaluate_qa(encoded_qa_pairs[i])

        return [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def _encode_qa(self, question, answer):
        if type(answer) is list:
            for a in answer:
                if a['correct']:
                    correct_answer = a['answer']
        else:
            correct_answer = answer
        return self.qae_tokenizer(
            text=question,
            text_pair=correct_answer,
            pad_to_max_length=True,
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        )

    def _evaluate_qa(self, encoded_qa_pair):
        output = self.qae_model(**encoded_qa_pair)
        return output[0][0][1]


def print_qa(qa_list, show_answers=True):
    for i in range(len(qa_list)):
        space = ' ' * int(np.where(i < 9, 3, 4)) # wider space for 2 digit q nums

        print('{}) Q: {}'.format(i + 1, qa_list[i]['question']))

        answer = qa_list[i]['answer']

        # print a list of multiple choice answers
        if type(answer) is list:

            if show_answers:
                print('{}A: 1.'.format(space),
                      answer[0]['answer'],
                      np.where(answer[0]['correct'], '(correct)', ''))
                for j in range(1, len(answer)):
                    print('{}{}.'.format(space + '   ', j + 1),
                          answer[j]['answer'],
                          np.where(answer[j]['correct'] == True, '(correct)', ''))

            else:
                print('{}A: 1.'.format(space),
                      answer[0]['answer'])
                for j in range(1, len(answer)):
                    print('{}{}.'.format(space + '   ', j + 1),
                          answer[j]['answer'])
            print('')

        # print full sentence answers
        else:
            if show_answers:
                print('{}A:'.format(space), answer, '\n')