# Language detector 

_

This is a document level language detector. Each document is first tokenized into a list of sentences, and the language detector identifies what language each sentence is written in. Given a document, I randomly sample x% of the sentences from the document. and return the most frequently detected language to be the language the document is written in. 

## Model 1: xlm-roberta-base-language-detection 
This is an [language detection model](https://huggingface.co/papluca/xlm-roberta-base-language-detection) available on huggingface which supports 20 languages. 

## Model 2: Character-level language model 
This method is an extension of one of my coursework where I developed a character-level n-gram language model for 3 languages. I extended this model to 53 languages train

ed on the [news crawl data](https://data.statmt.org/news-crawl/). 
During the inference time, the perplexity of each sentence under all the n-gram language models is computed and the language with the lowest perplexity score is chosen to be the language in a sentence written in. Details of the training setup and training data can be found under the directory ```ngram_lm```. 

## Model 3: bert-based language detection 
This language detection model is fine-tuned on [bert-multilingual-base-uncased](https://huggingface.co/bert-base-multilingual-uncased). It is trained on 34 languages from news crawl data](https://data.statmt.org/news-crawl/), plus 2 languages (Danish and Swedish) from [europarl corpus](https://www.statmt.org/europarl/) and achieves an accuracy of 0.95 and f1-score of 0.94 at the validation step. Another variant also supports 34 languages (without Swedish and Danish) and achieves accuracy and an f1-score of 0.97. 
Details of the training setup, experiment logs and training data can be found under the directory ```mbert-lang-id```.

