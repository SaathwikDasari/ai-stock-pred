// src/app/page.tsx
'use client';

import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, BarChart3, AlertCircle, Loader2 } from 'lucide-react';

interface StockData {
  date: string;
  price: number;
  type: 'historical' | 'predicted';
}

interface PredictionResult {
  ticker: string;
  data: StockData[];
  prediction: {
    trend: 'bullish' | 'bearish' | 'neutral';
    confidence: number;
    reasoning: string;
  };
}

export default function StockPredictor() {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');

  const fetchStockData = async () => {
  const response = await fetch('http://localhost:5000/api/receive_data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker }),
  });
  const data = await response.json();
  console.log(data);
  return data;
  }

  const handlePredict = async () => {
    if (!ticker.trim()) {
      setError('Please enter a stock ticker');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:5000/api/receive_data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "ticker": ticker.toUpperCase() }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }

      const result = await response.json();
      console.log(result)
      console.log(setPrediction(result));

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate prediction');
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'bullish': return 'text-green-400';
      case 'bearish': return 'text-red-400';
      default: return 'text-yellow-400';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <TrendingUp className="text-purple-400" size={40} />
            <h1 className="text-4xl font-bold text-white">AI Stock Predictor</h1>
          </div>
        </div>

        {/* Input Section */}
        <div className="max-w-md mx-auto mb-8">
          <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
            <div className="mb-4">
              <label className="block text-white font-medium mb-2">Stock Ticker</label>
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="e.g., AAPL, GOOGL, TSLA"
                className="w-full px-4 py-3 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-400 focus:bg-white/30 transition-all"
                onKeyPress={(e) => e.key === 'Enter' && handlePredict()}
              />
            </div>
            
            <button
              onClick={fetchStockData}
              disabled={loading}
              className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" size={20} />
                  Analyzing with AI...
                </>
              ) : (
                <>
                  <BarChart3 size={20} />
                  Predict Stock Price
                </>
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="max-w-2xl mx-auto mb-8">
            <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="text-red-400" size={24} />
              <p className="text-red-300">{error}</p>
            </div>
          </div>
        )}

        {/* Results */}
        {prediction && (
          <div className="max-w-6xl mx-auto">
            {/* Prediction Summary */}
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20 mb-8">
              <h2 className="text-2xl font-bold text-white mb-4">{prediction.ticker} Analysis</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <p className="text-gray-300 mb-1">Trend Prediction</p>
                  <p className={`text-xl font-bold capitalize ${getTrendColor(prediction.prediction.trend)}`}>
                    {prediction.prediction.trend}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-gray-300 mb-1">Confidence Level</p>
                  <p className="text-xl font-bold text-purple-400">
                    {(prediction.prediction.confidence * 100).toFixed(0)}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-gray-300 mb-1">Current Price</p>
                  <p className="text-xl font-bold text-white">
                    {formatPrice(prediction.data.find(d => d.type === 'historical')?.price || 0)}
                  </p>
                </div>
              </div>
              
              {/* AI Reasoning */}
              <div className="mt-6 p-4 bg-white/5 rounded-lg">
                <h3 className="text-lg font-semibold text-white mb-2">AI Analysis</h3>
                <p className="text-gray-300">{prediction.prediction.reasoning}</p>
              </div>
            </div>

            {/* Chart */}
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
              <h3 className="text-xl font-bold text-white mb-6">Price Prediction Chart</h3>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={prediction.data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="date" 
                      stroke="#9CA3AF"
                      tick={{ fontSize: 12 }}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      tick={{ fontSize: 12 }}
                      domain={['dataMin - 5', 'dataMax + 5']}
                      tickFormatter={formatPrice}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937', 
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#F3F4F6'
                      }}
                      formatter={(value: number) => [formatPrice(value), 'Price']}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      stroke="#8B5CF6"
                      strokeWidth={3}
                      dot={{ fill: '#8B5CF6', strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, fill: '#A855F7' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 flex items-center justify-center gap-8 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-blue-500 rounded"></div>
                  <span className="text-gray-300">Historical Data</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-purple-500 rounded"></div>
                  <span className="text-gray-300">AI Predictions</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

