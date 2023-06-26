## Approach

The problem was transformed into a classification task with three labels (low, medium, high) representing the relevance of an item to the customer. The classification was based on the order count of each item and the user's purchase history. The model takes user and item IDs, as well as additional features such as total quantity and total price, as input to predict the probability or relevance of a recommendation. The embeddings in the model capture hidden representations of users and items, while the biases capture user-specific and item-specific characteristics. By combining these embeddings, biases, and additional features through concatenation and using dropout and dense layers, the model learns a representation that can predict the probabilities of relevance for making recommendations.

## Challenging

1. High Imbalanced Order Count: The dataset had an imbalanced distribution of order counts. The three labels (low, medium, high) representing relevancy were affected by this imbalance. Approximately 50% of the dataset consisted of orders with a count of '1'.

2. Difficulty Implementing User-Item Relevancy: Creating a user-item relevancy feature that accurately captured the user's preference for an item proved to be challenging. The implementation of this feature was a complex task.

3. Commonly Purchased Items Together: Due to time constraints and a busy schedule, the feature of returning items that are commonly purchased together was not implemented. However, the intention was to explore and implement this feature using the existing model.

4. Overfitting: Initially, the model experienced significant overfitting due to the high imbalance in the order count. To address this issue, the number of orders was categorized into three ranges: low, medium, and high. This categorization helped mitigate overfitting:
   - Low: 1-2 orders
   - Medium: 2-5 orders
   - High: 5-59 orders

By implementing this relevancy label based on the order count categories, the overfitting problem was alleviated.

## Evaluation

The three models have slight variations in their inputs, outputs, and additional components like dropout layers, early stopping, and item text description. Here's a discussion on the differences between the three models based on the provided performance metrics:

1. First Model:
   - Inputs: user, item, total_quantity, total_price
   - Outputs: relevancy
   - Performance Metrics:

2. Second Model:
   - Inputs: user, item, total_quantity, total_price, description_input
   - Outputs: relevancy
   - Performance Metrics:

3. Third Model:
   - Inputs: user, item, total_quantity, total_price, description_input
   - Outputs: relevancy
   - Additional Components: dropout layer, early stopping, item text description
   - Performance Metrics:

## Discussion

The third model seems to perform relatively better on the medium and high classes compared to the first and second models. Here's a summary of the performance metrics for the medium and high classes:

Third Model:
- Medium Class:
  - Precision: 0.45
  - Recall: 0.43
  - F1-score: 0.44
- High Class:
  - Precision: 0.31
  - Recall: 0.34
  - F1-score: 0.33

The third model shows better precision, recall, and F1-score for both the medium and high classes compared to the first and second models. It indicates that the third model has a relatively better ability to predict and classify instances into these classes. However, it's important to note that the performance on these classes can still be further improved.

## Future Work

- To improve the performance and address the challenges faced, there are a few suggested strategies:
  - Relevance Metric: Enhance the relevance metric by incorporating additional factors such as the number of orders and the quantity of items in each order. By considering these factors, the relevance metric can capture a more comprehensive representation of user-item interactions, potentially helping to overcome the challenges posed by imbalanced data and overfitting.
  - Exploring Different Architectures and Techniques: Experiment with different model architectures and techniques. This could involve trying out various deep learning architectures, such as recurrent neural networks (RNNs) or convolutional neural networks (CNNs), to capture different aspects of the data and potentially improve performance. Additionally, consider employing techniques like regularization, ensemble methods, or advanced optimization algorithms to enhance the model's ability to learn and generalize from the data.

By implementing these strategies and iterating on the model design, it is possible to improve the performance, tackle imbalanced data, and mitigate overfitting. Remember to evaluate the results carefully, monitor the model's performance on different classes, and iterate as needed to achieve the desired outcomes.
