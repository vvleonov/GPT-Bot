from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

model_name = 'sberbank-ai/rugpt3large_based_on_gpt2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)

prompt = '''Пользователь: Привет. Как дела?
Бот: Здравствуйте! Все хорошо

Бот: Чем я могу Вам помочь?
Пользователь: У меня есть пара вопросов

Пользователь: Ты знаешь, какая завтра погода будет?
Бот: Хороший вопрос... Нет. Посмотри сам

Пользователь: Можешь рассказать что-нибудь?
Бот: Конечно. Вчера я съел 3 торта и не почувствовал

'''


def clear_text(text, encoded_prompt):
    """

    :param text: Decoded context and generated sequence
    :param encoded_prompt: Encoded context
    :return: Just generated and cleaned part of text
    """

    start_token = "<s>"
    stop_token = "</s>"
    text = text[len(tokenizer.decode(encoded_prompt[0])):]

    if start_token in text:
        text = text[text.find(start_token):]

    if stop_token in text:
        text = text[:text.find(stop_token)]

    if text.startswith('- '):
        text = text[1:]

    if '\n' in text:
        text = text[:text.index('\n')]

    text = text.strip()

    return text


def generate_output(context):
    """

    :param context: Combination of starting prompt and user input
    :return: Cleaned model answer
    """

    prompt_text = context + '\n'
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors='pt')
    encoded_prompt = encoded_prompt.to(device)
    out = model.generate(
        input_ids=encoded_prompt,
        max_length=150,
        temperature=0.90,
        top_k=20,
        top_p=0.8,
        do_sample=True,
        repetition_penalty=1.0,
        pad_token_id=0
    )

    text = list(map(tokenizer.decode, out))[0]
    generated = clear_text(text, encoded_prompt)

    return generated


while True:
    context = [prompt]
    question = input('Пользователь: ')
    if question:
        context.append('Пользователь: ' + question)
        context = '\n'.join(context)
        answer = generate_output(context)
        if answer:
            # context.append('Бот: ' + answer)
            print(answer)
        else:
            # del context[-1]
            print('zzz... Акхм, извини, я вздремнул. Так о чём это ты?')
