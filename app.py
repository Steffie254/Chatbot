from flask import Flask, render_template, request
import chatbot

app = Flask(__name__)


# To remove any cache data on reload.
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get')
def get_bot_response():
    userText = request.args.get('msg')
    print(userText)
    return chatbot.chatbot_response(str(userText))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=False)



