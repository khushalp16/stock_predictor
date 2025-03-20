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
    if (stockData.length < 6) return; // Need at least 6 days of data
    setLoading(true);

    // Prepare features and labels using all available data
    const features: number[][] = [];
    const labels: number[] = [];

    // Calculate moving averages and other features for each day
    for (let i = 5; i < stockData.length - 1; i++) {
      // Use last 5 days of data for features
      const last5Days = stockData.slice(i - 5, i);
      const currentDay = stockData[i];
      const nextDay = stockData[i + 1];
      
      // Skip if any required data is missing or invalid
      if (!currentDay || !nextDay || last5Days.some(day => !day)) continue;
      
      // Calculate price-based features
      const avgPrice = last5Days.reduce((sum, day) => sum + (day?.price || 0), 0) / 5;
      const priceChange = currentDay.price - last5Days[4].price;
      const priceChangePercent = (priceChange / last5Days[4].price) * 100;
      const priceVolatility = Math.sqrt(
        last5Days.reduce((sum, day) => sum + Math.pow(day.price - avgPrice, 2), 0) / 5
      );
      
      // Calculate volume-based features
      const avgVolume = last5Days.reduce((sum, day) => sum + (day?.volume || 0), 0) / 5;
      const volumeChange = currentDay.volume - last5Days[4].volume;
      const volumeChangePercent = (volumeChange / last5Days[4].volume) * 100;
      
      // Calculate trend features
      const priceTrend = last5Days.map((day, idx) => day.price).reduce((acc, price, idx, arr) => {
        if (idx === 0) return 0;
        return acc + (price > arr[idx - 1] ? 1 : -1);
      }, 0) / 4; // Normalize by number of comparisons
      
      const volumeTrend = last5Days.map((day, idx) => day.volume).reduce((acc, volume, idx, arr) => {
        if (idx === 0) return 0;
        return acc + (volume > arr[idx - 1] ? 1 : -1);
      }, 0) / 4;
      
      // Skip if any calculated features are NaN
      if (isNaN(avgPrice) || isNaN(avgVolume) || isNaN(priceChange) || isNaN(volumeChange) ||
          isNaN(priceChangePercent) || isNaN(volumeChangePercent) || isNaN(priceVolatility) ||
          isNaN(priceTrend) || isNaN(volumeTrend)) continue;
      
      features.push([
        1, // bias term
        avgPrice,
        avgVolume,
        priceChange,
        volumeChange,
        priceChangePercent,
        volumeChangePercent,
        priceVolatility,
        priceTrend,
        volumeTrend
      ]);
      
      // Label is 1 if price went up next day, 0 if down
      labels.push(nextDay.price > currentDay.price ? 1 : 0);
    }

    // Skip if we don't have enough training data
    if (features.length < 2) {
      setLoading(false);
      return;
    }

    // Train model
    const model = trainModel(features, labels);

    // Make prediction using the last 5 days of data
    const last5Days = stockData.slice(-5);
    const lastDay = stockData[stockData.length - 1];
    
    // Skip if any required data is missing or invalid
    if (!lastDay || last5Days.some(day => !day)) {
      setLoading(false);
      return;
    }
    
    // Calculate the same features for prediction
    const avgPrice = last5Days.reduce((sum, day) => sum + (day?.price || 0), 0) / 5;
    const priceChange = lastDay.price - last5Days[4].price;
    const priceChangePercent = (priceChange / last5Days[4].price) * 100;
    const priceVolatility = Math.sqrt(
      last5Days.reduce((sum, day) => sum + Math.pow(day.price - avgPrice, 2), 0) / 5
    );
    
    const avgVolume = last5Days.reduce((sum, day) => sum + (day?.volume || 0), 0) / 5;
    const volumeChange = lastDay.volume - last5Days[4].volume;
    const volumeChangePercent = (volumeChange / last5Days[4].volume) * 100;
    
    const priceTrend = last5Days.map((day, idx) => day.price).reduce((acc, price, idx, arr) => {
      if (idx === 0) return 0;
      return acc + (price > arr[idx - 1] ? 1 : -1);
    }, 0) / 4;
    
    const volumeTrend = last5Days.map((day, idx) => day.volume).reduce((acc, volume, idx, arr) => {
      if (idx === 0) return 0;
      return acc + (volume > arr[idx - 1] ? 1 : -1);
    }, 0) / 4;
    
    // Skip if any calculated features are NaN
    if (isNaN(avgPrice) || isNaN(avgVolume) || isNaN(priceChange) || isNaN(volumeChange) ||
        isNaN(priceChangePercent) || isNaN(volumeChangePercent) || isNaN(priceVolatility) ||
        isNaN(priceTrend) || isNaN(volumeTrend)) {
      setLoading(false);
      return;
    }
    
    const result = predict([
      1, // bias term
      avgPrice,
      avgVolume,
      priceChange,
      volumeChange,
      priceChangePercent,
      volumeChangePercent,
      priceVolatility,
      priceTrend,
      volumeTrend
    ], model);
    
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