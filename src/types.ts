export interface StockData {
  date: string;
  price: number;
  volume: number;
}

export interface PredictionResult {
  probability: number;
  prediction: 'Up' | 'Down';
}