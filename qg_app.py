'''
Streamlit demo of Question Generator
'''


import streamlit as st
from questiongenerator import QuestionGenerator


def st_write_qa(qa_list, show_answers=True):
    for i in range(len(qa_list)):
        space = ' ' * int(np.where(i < 9, 3, 4)) # wider space for 2 digit q nums

        st.write('{}) Q: {}'.format(i + 1, qa_list[i]['question']))

        answer = qa_list[i]['answer']

        # print a list of multiple choice answers
        if type(answer) is list:

            if show_answers:
                st.write('{}A: 1.'.format(space),
                      answer[0]['answer'],
                      np.where(answer[0]['correct'], '(correct)', ''))
                for j in range(1, len(answer)):
                    st.write('{}{}.'.format(space + '   ', j + 1),
                          answer[j]['answer'],
                          np.where(answer[j]['correct'] == True, '(correct)', ''))

            else:
                st.write('{}A: 1.'.format(space),
                      answer[0]['answer'])
                for j in range(1, len(answer)):
                    st.write('{}{}.'.format(space + '   ', j + 1),
                          answer[j]['answer'])
            st.write('')

        # print full sentence answers
        else:
            if show_answers:
                st.write('{}A:'.format(space), answer, '\n')


qg = QuestionGenerator()

text = st.text_area("Input text:")
num_questions = st.slider('Choose the number of question to generate:', 1, 20)
qa_list = qg.generate(text, num_questions=num_questions)
st_write_qa(qa_list)
