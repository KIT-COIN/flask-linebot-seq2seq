from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

from six.moves import xrange

import tensorflow as tf

import data_utils
import translate
import os
import numpy as np

app = Flask(__name__)

app.config['CHANNEL_ACCESS_TOKEN'] = 'BBRbzcN0GjBelHIlfA0QkwsCGqN5kNVJcH9m5kEO//OPT74Ml0i5YAjHeEWUHU1HmAUfsJ/7bn6mQ1v1yQQSTIkZBnCdDDTrCrpqV3jORXuEy2oiPUXsLSbgjd6LHz1kdFnvcJxIWbpj0qrrlXewiwdB04t89/1O/w1cDnyilFU='
app.config['CHANNEL_SECRET'] = '21a4d78b5cd16fe7b22580095a364185'

line_bot_api = LineBotApi(app.config['CHANNEL_ACCESS_TOKEN'])
handler = WebhookHandler(app.config['CHANNEL_SECRET'])

in_vocab_path = os.path.join(translate.FLAGS.data_dir,
                             "vocab_in.txt")
out_vocab_path = os.path.join(translate.FLAGS.data_dir,
                              "vocab_out.txt")

in_vocab, _ = data_utils.initialize_vocabulary(in_vocab_path)
_, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)


@app.route("/")
def say_hello():
    return "Hello"


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print(
            "Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    input_text = event.message.text

    with tf.Session() as sess:
        # print ("Hello!!")
        model = translate.create_model(sess, True)
        model.batch_size = 1

        # in_vocab_path = os.path.join(translate.FLAGS.data_dir,
        #                              "vocab_in.txt")
        # out_vocab_path = os.path.join(translate.FLAGS.data_dir,
        #                               "vocab_out.txt")

        # in_vocab, _ = data_utils.initialize_vocabulary(in_vocab_path)
        # _, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)

        sentence = translate.wakati(input_text)
        token_ids = data_utils.sentence_to_token_ids(sentence, in_vocab)

        bucket_id = min([b for b in xrange(len(translate._buckets))
                         if translate._buckets[b][0] > len(token_ids)])

        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)

        _, _, output_logits = model.step(
            sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

        outputs = [int(np.argmax(logit, axis=1))
                   for logit in output_logits]

        if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]

        # print("".join([rev_out_vocab[output] for output in outputs]))

        reply = "".join([rev_out_vocab[output] for output in outputs])

        line_bot_api.reply_message(event.reply_token, TextSendMessage(
            text=reply), )


if __name__ == "__main__":
    app.run()
