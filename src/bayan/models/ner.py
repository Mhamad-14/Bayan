"""Lab 3B: NER BIO-label alignment."""


def align_labels(word_ids, word_labels):
    """Align word-level NER labels with tokenizer subwords.

    Special tokens and non-first subword pieces receive -100 so
    they are ignored when computing the token-classification loss.
    """

    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:

        # Special tokens such as [CLS], [SEP], and padding
        if word_id is None:
            aligned_labels.append(-100)

        # First token/subword belonging to this word
        elif word_id != previous_word_id:
            aligned_labels.append(
                word_labels[word_id]
            )

        # Additional subword belonging to the same original word
        else:
            aligned_labels.append(-100)

        previous_word_id = word_id

    return aligned_labels