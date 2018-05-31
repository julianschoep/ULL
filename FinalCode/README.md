File notation model shorthands: 
 - CBOW model: cbow
 - Embed-Align: embed
 - Skip-gram: skipgram
 - Skip-gram, sampled once: min or minskip
 
 
log_\<model\>eval.txt --> logs of senteval output for the different models </br>
\<model\>_train.py --> file for preprocessing done for each specific model </br>
\<model\>_eval.py --> SentEval initiator, including Batcher and Prepare methods for specific models </br>

- stopwords --> stopwords actually used in paper </br>
- stopwords_strict --> not used </br>


