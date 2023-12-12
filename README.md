This repository includes programming and modeling codes for CSE 8803 - Deep Learning for Text class. It was completed while working with 3 batchmates: Anshit Verma, Manoj Parmar, and Samaksh Gulati.

Recently, a lot of research has shown that domain or task focused finetuning can improve the performance of large language models. Finetuning is a supervised methodology and requires labeled data. As human annotated labels are expensive to generate, we experiment with machine generated labels and how does the finetuned model perform using this data.
We experiment for 2 taks:
1. Multiclass classification task: Using conference title dataset
2. Question answering task: Using a subset of SQUAD dataset

Results show that finetuning BERT model for multiclass classification task generates comparable result with human annotated finetuned model. Though we see significant performance drop for a more complex task such as question asnwering.

You can find the link to the project report here: [Link](https://github.com/pchoudhary23/Efficacy-of-Machine-generated-annotations-in-finetuning-BERT-language-model/blob/main/DLT_Final_Report.pdf)
