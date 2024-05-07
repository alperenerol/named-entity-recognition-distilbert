from lxml import etree
from transformers import DistilBertTokenizerFast
import numpy as np

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

# Function to parse a XMI file
def parse_xmi(file_path):
    tree = etree.parse(file_path)
    root = tree.getroot()

    # Extract namespaces directly from the parsed XML
    namespaces = {key: value for key, value in root.nsmap.items() if key}

    # Extracting the original document text
    sofa_string = root.find(".//cas:Sofa", namespaces).attrib['sofaString']

    # Parsing sentences
    sentences = []
    for sentence in root.findall(".//type2:Sentence", namespaces):
        start = int(sentence.attrib['begin'])
        end = int(sentence.attrib['end'])
        text = sofa_string[start:end]
        sentences.append({'begin': start, 'end': end, 'text': text})

    # Parsing tokens
    tokens = []
    for token in root.findall(".//type2:Token", namespaces):
        start = int(token.attrib['begin'])
        end = int(token.attrib['end'])
        text = sofa_string[start:end]
        tokens.append({'begin': start, 'end': end, 'text': text})

    # Parsing entities and their types
    entities = []
    for entity in root.findall(".//custom:TextMiningAnnotation", namespaces):
        start = int(entity.attrib['begin'])
        end = int(entity.attrib['end'])
        entity_type = entity.attrib['EntityType']
        entities.append({'begin': start, 'end': end, 'EntityType': entity_type})

    return sentences, tokens, entities

# Function to replace NER tags with their corresponding integer values
def replace_ner_tags_with_integers(dataset_item):
    new_ner_tags = [label_to_id[tag] for tag in dataset_item['ner_tags']]
    return {
        'id': dataset_item['id'],
        'tokens': dataset_item['tokens'],
        'ner_tags': new_ner_tags
    }

# Function to assign BIO-Scheme NER tags to tokens
def assign_ner_tags(tokens, entities):
    # Sort tokens and entities by their beginning positions to ensure correct ordering
    tokens.sort(key=lambda x: x['begin'])
    entities.sort(key=lambda x: x['begin'])

    for token in tokens:
        token['ner_tag'] = 'O'  # Default tag
    for entity in entities:
        # Find tokens that fall within the current entity's range
        entity_tokens = [token for token in tokens if token['begin'] >= entity['begin'] and token['end'] <= entity['end']]
        if entity_tokens:
            # Mark the first token as B-entity
            entity_tokens[0]['ner_tag'] = 'B-' + entity['EntityType']
            # Mark subsequent tokens as I-entity
            for token in entity_tokens[1:]:
                token['ner_tag'] = 'I-' + entity['EntityType']
    return tokens

# Function to format the dataset
def format_dataset(sentences, tokens):
    dataset = []
    for sentence in sentences:
        sentence_tokens = [token for token in tokens if token['begin'] >= sentence['begin'] and token['end'] <= sentence['end']]
        dataset_item = {
            'id': str(sentence['begin']),  # Using sentence 'begin' as a unique ID; 
            #+100000, +300000, +600000, +900000 added to int(sentence['begin']) to make ids unique in combined train dataset # Using sentence 'begin' as a unique ID; +100000, +300000, +600000, +900000 added to sentence['begin'] to make ids unique in combined train dataset (to prevent id overlap)
            'tokens': [token['text'] for token in sentence_tokens],
            'ner_tags': [token['ner_tag'] for token in sentence_tokens]
        }
        dataset.append(dataset_item)
    return dataset

# Function to align labels and tokenized inputs
def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Directly use the label for the first sub-token of the word
                label_ids.append(label[word_idx])
            else:
                # For subsequent sub-tokens in the same word:
                if label_all_tokens:
                    # Change the label to 'I-' if it was a 'B-' label
                    if label[word_idx] % 2 == 1:  # Assuming odd labels are 'B-' tags
                        label_ids.append(label[word_idx] + 1)  # Convert 'B-' to 'I-' tag
                    else:
                        label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Function to extract entities from BIO-scheme to categories
def extract_entities(model_output):
    extracted_entities = []
    current_entity = None
    current_word = ""

    for token in model_output:
        entity_type = token['entity'].split('-')[-1]  # Get the entity type without the B-/I- prefix
        if token['entity'].startswith('B-') or current_entity != entity_type:
            # If starting a new entity or changing the entity type, save the current entity and start a new one
            if current_entity is not None:
                extracted_entities.append({'entity': current_entity, 'word': current_word})
            current_entity = entity_type
            current_word = token['word'].replace('##', '')  # Start a new word, removing BERT's subword prefix
        else:
            # If continuing the same entity, concatenate the word
            # Add a space if not a subword (does not start with ##), else concatenate directly
            separator = "" if token['word'].startswith('##') else " "
            current_word += separator + token['word'].replace('##', '')

    # Don't forget to add the last entity to the list
    if current_entity is not None:
        extracted_entities.append({'entity': current_entity, 'word': current_word})

    return extracted_entities

# Compute metrics for evaluation and training
def compute_metrics(eval_preds):
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [id_to_label[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
      [id_to_label[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
       for prediction, label in zip(pred_logits, labels)
   ]
    results = metric.compute(predictions=predictions, references=true_labels)

    return {
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"],
          "accuracy": results["overall_accuracy"],
  }