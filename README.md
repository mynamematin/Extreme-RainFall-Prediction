# Rainfall Prediction using ConvLSTM

![Project Image](project_image.png) (Optional: You can add an image that represents your project here)

## Description

This repository contains the code for a Rainfall Prediction model using Convolutional Long Short-Term Memory (ConvLSTM) algorithm. The model is designed to forecast rainfall based on historical rainfall and sea surface temperature (SST) data.

The ConvLSTM model is a deep learning algorithm that learns the temporal dependencies in sequential data, making it suitable for spatiotemporal analysis like predicting weather patterns.

## Key Features

- Utilizes ConvLSTM to capture temporal dependencies in rainfall and SST data.
- Predicts rainfall values for future time periods.
- Identifies extreme rainfall events during the monsoon season.
- Provides insights for climate science research and policy interventions.

## Installation

Clone this repository to your local machine:
git clone https://github.com/your-username/your-repo-name.git


## Requirements

- Python 3
- Required Python packages: pandas, numpy, tensorflow (or tensorflow-gpu), matplotlib

To install the required packages, run: pip install pandas numpy tensorflow matplotlib


## Usage

1. Preprocess the data:
   - Apply bandpass filter to the satellite data.
   - Remove intraseasonal variabilities below 30 days and above 60 days.
   - Select data for specific months and normalize the data.
   - Perform exponential space transformation.

2. Train the model:
   - Set up the input and output sequences for training.
   - Train the ConvLSTM model using historical rainfall and SST data.

3. Evaluate the model:
   - Compare the predicted rainfall values with actual values.
   - Assess the accuracy and reliability of the predictions.

4. Make predictions:
   - Use the trained model to forecast rainfall for future time periods.

## Contributing

Contributions are welcome! If you have any ideas or improvements for the project, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

Special thanks to [Any collaborators or resources you want to acknowledge].



