from flask import Flask, render_template, request
from assistant import generate_summary, generate_questions, create_mcqs, translate_to_urdu

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/process', methods=['POST'])
def process():
    story_text = request.form['story_text']
    options = request.form.getlist('generate_options')
    
    summary = questions = mcqs = story_text_urdu = None

    if 'summary' in options:
        summary = generate_summary(story_text)
    if 'questions' in options:
        questions = generate_questions(story_text)
    if 'mcqs' in options:
        mcqs = create_mcqs(story_text)
    if 'story_text_urdu' in options:
        story_text_urdu = translate_to_urdu(story_text)

    return render_template('result.html', summary=summary, questions=questions, mcqs=mcqs, story_text_urdu=story_text_urdu)

if __name__ == '__main__':
    app.run(debug=True)
