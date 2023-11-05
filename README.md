# Sentiment-Analysis-Feature-Extraction-
In this work, we focus on GREEN AI considering CO2 emission. In this work, we investigate the relationship between performance, resource use, and environmental impact in sentiment analysis by measuring accuracy, end-to-end runtime, memory usage, energy expenditure, and generated CO2 estimates for a broad range of models. We also use different feature extraction techniques (BoW, TF-IDF, CBoW, FastText, DWE (FastText + GloVe), CNN, LSTM, RoBERTa, RoBERTa + FastText). Also, we use different Large Language Models (LLMs) (DistilBERT, ALBERT, ROBERTa) to perform sentiment analysis and explore how it affects Accuracy vs. computational Resources (especially CO2).  We find that while a fine-tuned LLM (RoBERTa) achieves the best accuracy, some alternate configurations (non-finetuned RoBERTa+FastText (feature extraction) with SVM (as a classifier)) provide huge (up to 24,283 times) resource savings (CO2) for a marginal (<1%) loss in accuracy.

This work is accepted and presented at ''The 11th International Workshop on Natural Language Processing for Social Media In conjunction with IJCNLP-AACL 2023, Bali, Indonesia''.

The paper title is 'Efficient Sentiment Analysis: A Resource-Aware Evaluation of Feature Extraction Techniques, Ensembling, and Deep Learning Models' and is available at: https://arxiv.org/abs/2308.02022

If you use this work, please cite:



@misc

{kamruzzaman2023efficient,

      title={Efficient Sentiment Analysis: A Resource-Aware Evaluation of Feature Extraction Techniques, Ensembling, and Deep Learning Models}, 
      
      author={Mahammed Kamruzzaman and Gene Louis Kim},
      
      year={2023},
      
      eprint={2308.02022},
      
      archivePrefix={arXiv},
      
      primaryClass={cs.CL}
      
}
