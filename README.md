# Movie Review Classifier

This application predicts the sentiment (positive or negative) of a movie review using SimpleRNN model, trained on the IMDb dataset.

## Project Structure
- `model.h5`: The trained RNN model used for prediction.
- `app.py`: The main Streamlit application file where the review classification logic is implemented.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or above
- Required Python libraries: 
  - Streamlit
  - TensorFlow
  - Scikit-learn
  - Pandas
  - Numpy

Libraries Installation:

```bash
pip install -r requirements.txt
```

Running the streamlit app:

```bash
streamlit run app.py
```
