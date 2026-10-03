# Data card

## Financial PhraseBank
Source: [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank), pinned revision `8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`.

License: [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/); noncommercial restrictions apply to data and derived redistributions. This repository redistributes no raw benchmark sentences. Citation: Malo, Sinha, Korhonen, Wallenius and Takala (2014), *Good debt or bad debt: Detecting semantic orientations in economic texts*.

Configuration: `sentences_allagree`, originally 2,264 investor-oriented financial sentences with complete annotator agreement. Labels are negative, neutral, positive. There is no official split. The script deduplicates normalized text, shuffles within label with seed 42, and allocates 70% train, 15% validation, remainder test. Exact post-cleaning counts are generated in metrics metadata. Splits are frozen before evaluation. No tuning or fitting is performed.

Bias: financial-news English, investor interpretation of sentiment, expert annotators from a narrow institutional setting. This is a domain-shift test for the social-media sentiment model, not a universal sentiment benchmark.

## DAIR-AI Emotion
Source: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), revision `cab853a1dbdf4c42c2b3ef2173804746df8825fe`, configuration `split`.

The card’s license field is **other**, with the explicit instruction **educational and research purposes only**. It is not claimed to be Apache, MIT or Creative Commons licensed. No raw examples are redistributed. Citation: Saravia et al. (2018), *CARER: Contextualized Affect Representations for Emotion Recognition*, EMNLP, DOI 10.18653/v1/D18-1404.

Original split: train 16,000; validation 2,000; test 2,000. Six labels: sadness, joy, love, anger, fear, surprise. Normalize case and non-word characters for duplicate hashing, keep train first, then remove cross-split/within-split duplicates from validation/test. Exact removal counts are recorded; the resulting cleaned set differs from unfiltered upstream reports. No test-set tuning. The pretrained classifier’s upstream dataset usage must be considered when interpreting reproduction results.

Bias: social-media English and automated distant-supervision labels; class imbalance and ambiguous labels. Original users did not consent to use as clinical evidence. Do not infer mental health or protected attributes.

## Local handling
`data/cache/` is ignored. Scripts download from official pinned sources and never execute dataset-provided code. Financial PhraseBank ZIP is read in memory, without extracting untrusted paths. Track only source metadata, aggregate metrics and error indices. Included sample CSV is newly authored illustrative text, not benchmark data.

Normalized duplicate checks identify likely textual leakage, not paraphrase or pretraining overlap. Model and data licenses are distinct from the MIT code license. TweetEval is excluded because its official card declares an unknown dataset license.
