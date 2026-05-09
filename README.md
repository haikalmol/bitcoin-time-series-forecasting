# Bitcoin Time Series Forecasting with Deep Learning

## 📋 Project Overview

This project implements an advanced **Bitcoin price forecasting system** using deep learning models. The system predicts Bitcoin prices for the next 24 hours using historical data with a 120-hour input window. Two models are compared: a baseline LSTM with Multi-Head Attention and a sophisticated Sequence-to-Sequence (Seq2Seq) architecture with custom attention mechanisms.

## 🎯 Project Objectives

- **Multi-step Time Series Forecasting**: Predict Bitcoin prices 24 hours ahead
- **Advanced Deep Learning Architecture**: Implement custom layers and attention mechanisms
- **Comparative Analysis**: Compare baseline vs. state-of-the-art models
- **Real-world Application**: Handle actual cryptocurrency market data

## 📊 Dataset

**Source**: Bitcoin3.csv (6.4MB)
**Features Used**:
- `Close`: Bitcoin closing price (USDT)
- `Volume USDT`: Trading volume
- `RSI`: Relative Strength Index
- `Rolling_Mean_24`: 24-hour rolling average (engineered feature)

**Data Split**:
- Training: 70% (37,188 samples)
- Validation: 20% (10,625 samples)  
- Testing: 10% (5,314 samples)

## 🏗️ Model Architecture

### 1. Baseline Model (LSTM + Multi-Head Attention)
- **Input Layer**: 120 timesteps × 4 features
- **LSTM Layer**: 64 hidden units with tanh activation
- **Multi-Head Attention**: 4 heads, 16 key dimensions
- **Global Average Pooling**: Reduces sequence to single vector
- **Output Layer**: Linear activation for 24-hour predictions

### 2. Advanced Seq2Seq Model (State-of-the-Art)
- **Encoder**: LSTM with 64 hidden units, returns sequences and states
- **Decoder**: LSTM with Multi-Head Attention mechanism
- **Custom Components**:
  - Custom Multi-Head Attention Layer
  - Custom Dense Layer with configurable activation
  - Scheduled Sampling for training stability
  - Weighted MAE loss function

## 🔧 Technical Implementation

### Custom Layers

#### Multi-Head Attention
```python
class CustomMultiHeadAttention(Layer):
    - 4 attention heads
    - 16-dimensional keys
    - Scaled dot-product attention
    - Full implementation from scratch
```

#### Custom Dense Layer
```python
class CustomDenseLayer(Layer):
    - Configurable units and activation
    - Custom weight initialization
    - Serializable configuration
```

### Training Strategy

#### Baseline Model
- **Optimizer**: Adam (default learning rate)
- **Loss**: Mean Squared Error
- **Metrics**: MAE
- **Early Stopping**: Patience = 5 epochs
- **Max Epochs**: 30

#### Seq2Seq Model (Advanced)
- **Optimizer**: Adam (learning rate = 0.0001)
- **Loss**: Custom Weighted MAE (higher penalty for distant predictions)
- **Training Method**: Custom training loop with GradientTape
- **Scheduled Sampling**: Teacher forcing ratio decays from 1.0 to 0.0
- **Advanced Callbacks**: Manual Early Stopping + Learning Rate Decay
- **Max Epochs**: 50

## 📈 Performance Results

### Model Comparison (Test Set)

| Model | MAE (Scaled) | MAE (USDT) | RMSE (USDT) |
|-------|-------------|------------|-------------|
| Baseline MHA | 0.3090 | $20,230.21 | $20,238.66 |
| **Seq2Seq + MHA (SOTA)** | **0.0134** | **$877.75** | **$1,094.63** |

### Key Findings
- **Seq2Seq model outperforms baseline by ~23x in MAE**
- **Significant improvement in prediction accuracy**
- **Better handling of long-term dependencies**

## 🚀 Usage Instructions

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Project
1. **Jupyter Notebook** (Recommended):
   ```bash
   jupyter notebook Haikal_Submission_Akhir_DLTM.ipynb
   ```

2. **Python Script**:
   ```bash
   python Haikal_Submission_Akhir_DLTM.py
   ```

### Model Files
- `model_baseline_LSTM.keras`: Baseline model architecture + weights
- `model_encoder_Seq2Seq.keras`: Seq2Seq encoder model
- `model_decoder_Seq2Seq.keras`: Seq2Seq decoder model
- `best_encoder.weights.h5`: Best encoder weights during training
- `best_decoder.weights.h5`: Best decoder weights during training

## 📁 Project Structure

```
├── Haikal_Submission_Akhir_DLTM.ipynb    # Main Jupyter notebook
├── Haikal_Submission_Akhir_DLTM.py       # Python script version
├── Bitcoin3.csv                          # Dataset (6.4MB)
├── requirements.txt                      # Dependencies
├── model_baseline_LSTM.keras             # Baseline model
├── model_encoder_Seq2Seq.keras           # Seq2Seq encoder
├── model_decoder_Seq2Seq.keras           # Seq2Seq decoder
├── best_encoder.weights.h5               # Best encoder weights
├── best_decoder.weights.h5               # Best decoder weights
└── README.md                             # This file
```

## 🔬 Key Features

### Data Preprocessing
- **Min-Max Scaling**: Normalizes all features to [0,1] range
- **Sequential Split**: Prevents data leakage in time series
- **Feature Engineering**: 24-hour rolling mean for trend context

### Exploratory Data Analysis
- **Correlation Matrix**: Feature relationship analysis
- **Time Series Decomposition**: Trend, seasonal, and residual components
- **ACF/PACF Analysis**: Autocorrelation patterns for lag selection

### Advanced Training Techniques
- **Scheduled Sampling**: Gradual transition from teacher forcing to autonomous prediction
- **Custom Loss Function**: Weighted MAE emphasizing long-term accuracy
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Adaptive learning rate decay

## 🎯 Model Evaluation

### Metrics Used
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
- **Scaled vs Real Values**: Performance in both normalized and original scales

### Visualization
- **24-hour Prediction Comparison**: Actual vs. Predicted prices
- **Training History**: Loss curves over epochs
- **Attention Weights**: (Optional) Visualize attention patterns

## 🛠️ Dependencies

```
google_play_scraper==1.2.7
matplotlib==3.10.8
numpy==2.4.2
pandas==3.0.1
scikit_learn==1.8.0
seaborn==0.13.2
tensorflow==2.20.0
xgboost==3.2.0
```

## 📝 Implementation Notes

### Reproducibility
- **Random Seeds**: Fixed seeds for TensorFlow (42) and NumPy (42)
- **Deterministic Training**: Consistent results across runs

### Performance Optimization
- **tf.data Pipeline**: Efficient data loading and preprocessing
- **@tf.function**: Graph compilation for faster training
- **Batch Processing**: Optimized for GPU acceleration

### Model Serialization
- **Keras Format**: Native .keras format for easy deployment
- **Custom Layers**: Properly implemented `get_config()` methods
- **Weight Management**: Separate weight files for best models

## 🏆 Achievements

1. **Custom Implementation**: Built Multi-Head Attention from scratch
2. **Advanced Architecture**: Seq2Seq with attention mechanism
3. **Superior Performance**: 23x improvement over baseline
4. **Production Ready**: Proper model saving and loading
5. **Comprehensive Evaluation**: Multiple metrics and visualizations

## 🔮 Future Improvements

- **Additional Features**: Technical indicators, sentiment analysis
- **Ensemble Methods**: Combine multiple models
- **Hyperparameter Optimization**: Automated tuning
- **Real-time Prediction**: Live data integration
- **Multi-asset Forecasting**: Extend to other cryptocurrencies

## 👨‍💻 Author

**Haikal** - Deep Learning Practitioner
- Final Project Submission for Deep Learning Course
- Advanced Time Series Forecasting Implementation
- Custom Architecture Development

---

*This project demonstrates advanced deep learning techniques for financial time series forecasting, achieving state-of-the-art performance through custom architecture design and sophisticated training strategies.*
