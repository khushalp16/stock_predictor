# Stock Market Predictor

A React-based web application that predicts stock price movements using machine learning. This tool analyzes historical stock data from NASDAQ and provides predictions about future price movements with confidence levels.

## Features

- 📈 Interactive stock price and volume visualization
- 🤖 Machine learning-based price movement prediction
- 📊 Support for historical NASDAQ stock data
- 🎯 Confidence level indicators for predictions
- 📱 Responsive design with modern UI

## Tech Stack

- React + TypeScript
- Tailwind CSS for styling
- Recharts for data visualization
- Custom machine learning implementation
- Vite for build tooling

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock_market_predictor.git
cd stock_market_predictor
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Start the development server:
```bash
npm run dev
# or
yarn dev
```

4. Open your browser and navigate to `http://localhost:5173`

## Usage

1. Download historical stock data from [NASDAQ](https://www.nasdaq.com) in CSV format
2. Upload the CSV file through the application interface
3. View the interactive chart showing price and volume data
4. Click "Predict Next Movement" to get the prediction
5. Review the prediction and confidence level

## Data Format

The application expects CSV files with the following columns:
- Date
- Close Price
- Volume
- Open Price
- High Price
- Low Price

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with React and TypeScript
- Data visualization powered by Recharts
- Styling with Tailwind CSS 
