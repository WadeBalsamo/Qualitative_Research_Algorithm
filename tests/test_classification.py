import numpy as np
from mentalbert_sentence_aqua import classification

def test_classify_sentences_shape():
    class DummyModel:
        def __init__(self):
            self.device = 'cpu'
            import torch
            self.classifier = lambda x: torch.zeros((x.shape[0], 3))
    embeddings = np.zeros((5, 10))
    label_names = ['a', 'b', 'c']
    results = classification.classify_sentences(embeddings, DummyModel(), label_names)
    assert len(results) == 5
    for r in results:
        assert 'label' in r and 'confidence' in r and 'probs' in r

def test_aggregate_community_labels():
    partition = {0: 1, 1: 1, 2: 2}
    sentence_labels = [
        {'label': 'a'}, {'label': 'a'}, {'label': 'b'}
    ]
    comm_labels = classification.aggregate_community_labels(partition, sentence_labels)
    assert comm_labels[1] == 'a'
    assert comm_labels[2] == 'b'
