import React, { useState, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { TrendingUp, TrendingDown, BarChart2 } from 'lucide-react';
import { StockData, PredictionResult } from '../types';
import { trainModel, predict } from '../utils/ml';

const StockPredictor: React.FC = () => {
  const [stockData, setStockData] = useState<StockData[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const rows = text.split('\n');
      const parsedData: StockData[] = rows
        .slice(1) // Skip header row
        .filter(row => row.trim())
        .map(row => {
          const [date, closePrice, volume, open, high, low] = row.split(',');
          // Remove dollar sign and parse price
          const cleanPrice = closePrice.replace('$', '').trim();
          // Parse volume as integer
          const cleanVolume = volume.replace(/,/g, '').trim();
          return {
            date,
            price: parseFloat(cleanPrice),
            volume: parseInt(cleanVolume, 10)
          };
        });
      setStockData(parsedData);
    };
    reader.readAsText(file);
  };

  const makePrediction = useCallback(() => {
    if (stockData.length < 2) return;
    setLoading(true);

    // Prepare features and labels
    const features: number[][] = [];
    const labels: number[] = [];

    for (let i = 1; i < stockData.length; i++) {
      const prevDay = stockData[i - 1];
      const currentDay = stockData[i];
      
      features.push([
        1, // bias term
        prevDay.price,
        prevDay.volume
      ]);
      
      labels.push(currentDay.price > prevDay.price ? 1 : 0);
    }

    // Train model
    const model = trainModel(features, labels);

    // Make prediction using the last day's data
    const lastDay = stockData[stockData.length - 1];
    const result = predict([1, lastDay.price, lastDay.volume], model);
    
    setPrediction(result);
    setLoading(false);
  }, [stockData]);

  const formatVolume = (value: number) => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    return value.toString();
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
        <div className="flex items-center mb-6">
          <BarChart2 className="w-8 h-8 text-blue-600 mr-3" />
          <h1 className="text-2xl font-bold text-gray-800">Stock Price Predictor</h1>
        </div>

        <div className="mb-8">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Upload CSV (<a href="https://www.nasdaq.com" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800">Nasdaq</a> stock data only)
          </label>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>

        {stockData.length > 0 && (
          <div className="mb-8 overflow-x-auto">
            <LineChart 
              width={800} 
              height={400} 
              data={stockData}
              margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                interval={Math.floor(stockData.length / 10)} // Show fewer x-axis labels
              />
              <YAxis 
                yAxisId="left"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `$${value.toFixed(2)}`}
              />
              <YAxis 
                yAxisId="right"
                orientation="right"
                tick={{ fontSize: 12 }}
                tickFormatter={formatVolume}
              />
              <Tooltip 
                formatter={(value: number, name: string) => [
                  name === 'price' ? `$${value.toFixed(2)}` : formatVolume(value),
                  name
                ]}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <Legend />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="price" 
                stroke="#2563eb" 
                strokeWidth={2}
                dot={false}
                name="Price"
              />
              <Line 
                yAxisId="right"
                type="monotone" 
                dataKey="volume" 
                stroke="#7c3aed" 
                strokeWidth={2}
                dot={false}
                name="Volume"
              />
            </LineChart>
          </div>
        )}

        <button
          onClick={makePrediction}
          disabled={loading || stockData.length < 2}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Processing...' : 'Predict Next Movement'}
        </button>

        {prediction && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                {prediction.prediction === 'Up' ? (
                  <TrendingUp className="w-6 h-6 text-green-500 mr-2" />
                ) : (
                  <TrendingDown className="w-6 h-6 text-red-500 mr-2" />
                )}
                <span className="text-lg font-semibold">
                  Predicted Movement: {prediction.prediction}
                </span>
              </div>
              <span className="text-sm text-gray-600">
                Confidence: {prediction.probability}%
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockPredictor;