# Sentiment_Analysis_Naive_Bayes
Creating from scratch a Naive Bayes Classifier in order to perform Sentiment Analysis on a dataset of many customers' reviews. 

The dataset can be found in the file 'all_sentiment_shuffled.txt'. 

Classifying customer reviews as "positive" or "negative" with the use of a Naive Bayes Classifier that I created from scratch. 

Along with the explanatory comments that you may find in the Python code, below are
some further comments and the answers to the descriptive parts of the assignment.

First of all, before creating the train_nb function we split the dataset into training and
validation and we created the vocabulary of all unique words, as it can be seen in the
program.

Next, in the train_nb function we compute the likelihood dictionary of all words being in a
negative or positive labeled review. We use the smoothing parameter alpha = 1.

Moving on, we created the function score_doc_label that computes the score of a review
both being positive or negative labeled according to its word’s likelihoods and also
according to the prior probabilities (of a document/review being positive or negative). As you
can see in the function, we set a limitation for the score of a review to not become smaller
than 10^(-305), which is a pretty small number for a probability. That is because Python
could not successfully handle very very small values and instead it was converting them to
0, thus making the log of that probability being -infinity.

In the function classify_nb, a document is being classified as positive or negative according
to which of the two scores is higher (which logarithmic probability of the two is higher).
The function classify_documents takes all of the documents and classifies them by calling
the previous functions and then it returns a list that contains all of the guessed labels for the
documents.

Continuing, we compute the accuracy of our classifier on the validation set. We get an accuracy of
nearly 80% (79,6%), which we believe is normal for this kind of naive bayes classifier.

After that, we compute the precision and the recall in order to compute the F1 score. The F1
score turned out to be almost the same with the accuracy but just slightly lower. It was
78,47%.

Regarding the Cross Validation, we firstly implemented the code for K-fold cross validation
and we run it for a number of folds N = 10. The results for the accuracy and the F1 score
were a bit lower than those that we had before the cross validation. More specifically, the
accuracy was 78.8% and the F1 score was 76.49%. We can tell that it was slightly lower
because in those many iterations the classifier was not able to predict the same or higher
percentages of labels of the given folds of documents than those that it could with the
previous validation dataset. Below you can see the code with the results.

After that, we computed the accuracy and the F1 score with the Leave-one-out Cross
Validation method. We stopped after 100 iterations as it was mentioned in the description of
the assignment. The accuracy with this validation method was 79.0%, while the F1 score
was 81.1%. This time the two metrics were very close to each other. There are slightly
better results in the leave-one out validation method.

The accuracy of the leave one out was higher in both f score and accuracy. The iterations in
this method were many more than in the previous and now each time the validation set was
consisting of just one document. Therefore, if in one iteration the document was classified
correctly, then the accuracy of that iteration was 100%, while if it was misclassified, the
accuracy of the iteration was 0%. Thus, the final accuracy was calculated as the mean of all
of the 100 accuracies and it was different from the previous two validation methods.
