import tensorflow as tf
import numpy as np
import json

ft_file = './data/e2e/train.jsonl'
ft_samples = []
with open(ft_file, 'r') as reader:
    for line in reader:
        items = json.loads(line.strip())
        context = items['context']
        completion = items['completion']
        ft_samples.append([context, completion])

def padding_tokens(tokens, max_seq_length, pad_token, direct, max_context_length=0):

    if max_context_length == 0:
        max_context_length = max_seq_length

    if len(tokens) > max_context_length:
        if direct > 0:
            pad_tokens = tokens[:max_context_length]
        else:
            pad_tokens = tokens[-max_context_length:]
    else:
        pad_tokens = tokens
    token_len = len(pad_tokens)
    pad_tokens = pad_tokens + [pad_token for _ in range(max_seq_length - token_len)]
    return pad_tokens, token_len

def parse(example, max_seq_length=512):
    context = example[0]
    completion = example[1]

    conditions = context
    _input, _input_len = padding_tokens(conditions + completion, max_seq_length, 0, 1)

    pad_targets = context + completion
    _target, _ = padding_tokens(pad_targets[1:], max_seq_length, 0, 1)
    _msk = [0.0] * (len(conditions) - 1) + [1.0] * (_input_len - len(conditions))
    _msk, _ = padding_tokens(_msk, max_seq_length, 0.0, 1)

    output = {}
    
    _query, _query_len = padding_tokens(
        conditions, max_seq_length, 0, -1, 
        max_context_length = max_seq_length
    )
    output["query"] = np.array(_query, dtype=np.int64)
    output["query_len"] = np.array(_query_len, dtype=np.int64)

    output["input"] = np.array(_input, dtype=np.int64) 
    output["target"] = np.array(_target, dtype=np.int64) 

    output["mask"] = np.array(_msk, dtype=np.float32)

    return output

chunks = []
from tqdm import tqdm
with tf.io.TFRecordWriter("e2e_train.tfrecord") as file_writer:
    for example in tqdm(ft_samples):
        output = parse(example)
        record_bytes = tf.train.Example(features=tf.train.Features(feature={
            "input": tf.train.Feature(int64_list=tf.train.Int64List(value=output['input'])),
            "target": tf.train.Feature(int64_list=tf.train.Int64List(value=output['target'])),
            "mask": tf.train.Feature(float_list=tf.train.FloatList(value=output['mask']))
        })).SerializeToString()
        file_writer.write(record_bytes)
