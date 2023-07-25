# Rainfall Prediction using ConvLSTM

![Project Image](MonsoonPred2017(1).png) 
## Description


This repository contains the code for my MTech project, Rainfall Prediction using Convolutional Long Short-Term Memory (ConvLSTM) algorithm. The project was done in collaboration with Dr. Roxy Mathew Koll at the Indian Institute of Tropical Meteorology (IITM).

The aim of this project was to develop a model that can accurately predict rainfall based on historical rainfall and sea surface temperature (SST) data. The ConvLSTM model is a deep learning algorithm that captures temporal dependencies in sequential data, making it well-suited for spatiotemporal analysis like weather forecasting.

The project explores the use of ConvLSTM to forecast rainfall values for future time periods and identify extreme rainfall events during the monsoon season. The predictions obtained from the ConvLSTM model can have important implications for climate science research and inform policy interventions to mitigate the impacts of extreme weather phenomena.

The project serves as a valuable contribution to the domain of climate science and showcases the potential of deep learning techniques in addressing complex spatiotemporal prediction tasks.

## Key Features

- Utilizes ConvLSTM to capture temporal dependencies in rainfall and SST data.
- Predicts rainfall values for future time periods.
- Identifies extreme rainfall events during the monsoon season.
- Provides insights for climate science research and policy interventions.

## Installation

Clone this repository to your local machine:
git clone https://github.com/mynamematin/Extreme-RainFall-Prediction.git


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

This project is licensed under the [@matin](LICENSE).

---
*Author:* [Matin Ahmed]
*Email:* [matinahmed000@gmail.com]


