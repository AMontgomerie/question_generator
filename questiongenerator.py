import os
import sys
import math
import numpy as np
import torch
import spacy
import re
import en_core_web_sm
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import BertTokenizer, BertForSequenceClassification

class QuestionGenerator():

    def __init__(self, model_dir=None):
        QG_PRETRAINED = 't5-small'
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
        self.qg_model = T5ForConditionalGeneration(config) #.from_pretrained(QG_PRETRAINED)
        self.qg_model.resize_token_embeddings(len(self.qg_tokenizer))

        if model_dir:
            checkpoint = torch.load(os.path.join(model_dir, QG_FINETUNED_MODEL))
        else:
            checkpoint = torch.load(os.path.join(sys.path[0], QG_FINETUNED_MODEL))

        self.qg_model.load_state_dict(checkpoint['model_state_dict'])
        self.qg_model.to(self.device)

        self.qa_evaluator = QAEvaluator(model_dir)

    def generate_questions(self, article, use_evaluator=True, num_questions=None):
        print("\nGenerating questions...")
        qg_inputs, qg_answers = self.generate_qg_inputs(article, use_NER=True)
        generated_questions = self.generate_questions_from_inputs(qg_inputs)

        message = "{} questions doesn't match {} answers".format(
            len(generated_questions),
            len(qg_answers))
        assert len(generated_questions) == len(qg_answers), message

        if use_evaluator:
            print("\nEvaluating questions...")
            encoded_qa_pairs = self.qa_evaluator.encode_qa_pairs(generated_questions, qg_answers)
            scores = self.qa_evaluator.get_scores(encoded_qa_pairs)
            if num_questions:
                self._print_ranked_qa_pairs(generated_questions, qg_answers, scores, num_questions)
            else:
                self._print_ranked_qa_pairs(generated_questions, qg_answers, scores)

        else:
            self._print_all_qa_pairs(generated_questions, qg_answers)

    def generate_qg_inputs(self, text, use_NER=False):
        sentences = self._split_text(text)
        inputs = []
        answers = []
        if use_NER:
            prepped_inputs, prepped_answers = self._prepare_qg_inputs_NER(sentences)
            inputs.extend(prepped_inputs)
            answers.extend(prepped_answers)
        prepped_inputs, prepped_answers = self._prepare_qg_inputs(sentences, text)
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

    def _prepare_qg_inputs_NER(self, sentences):
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
                    inputs_from_text.append(qg_input)
                    answers_from_text.append(str(entity))

        return inputs_from_text, answers_from_text

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

    def _print_ranked_qa_pairs(self, generated_questions, qg_answers, scores, num_questions=10):
        if num_questions > len(scores):
            num_questions = len(scores)
            print("\nWas only able to generate {} questions. For more questions, please input a longer text.".format(num_questions))
        print("\nTop {} generated questions:".format(num_questions))
        for i in range(num_questions):
            index = scores[i]
            q_num = i + 1
            print('{}) Q: {}'.format(q_num, generated_questions[index].split('?')[0] + '?'))
            print('{}A:'.format(' ' * int(np.where(q_num < 10, 3, 4))), qg_answers[index], '\n')

    def _print_all_qa_pairs(self, generated_questions, qg_answers):
        print("\nAll generated questions:")
        for i in range(len(generated_questions)):
            q_num = i + 1
            print('{}) Q: {}'.format(q_num, generated_questions[i].split('?')[0] + '?'))
            print('{}A:'.format(' ' * int(np.where(q_num < 10, 3, 4))), qg_answers[i], '\n')

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
            checkpoint = torch.load(os.path.join(sys.path[0], QAE_FINETUNED_MODEL))

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
        return self.qae_tokenizer(
            text=question,
            text_pair=answer,
            pad_to_max_length=True,
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        )

    def _evaluate_qa(self, encoded_qa_pair):
        output = self.qae_model(**encoded_qa_pair)
        return output[0][0][1]