1. The models should be able to communicate with each other, so if the 2 models have different tokenizers, you could truncate and pad the inputs so that they are compatible. You could also use a common tokenizer for post-processing.

2. I think constrative decoding is used due to its ability to debug large language models (LLMs). Additionally, it seems to show improved reasoning in LLMs and reduces the computing power needed.

Sources:
https://openreview.net/forum?id=SzV37yefM4
