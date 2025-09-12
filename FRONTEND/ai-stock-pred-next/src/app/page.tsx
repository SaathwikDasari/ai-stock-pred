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

  // Mock data generator for demonstration
  const generateMockData = (ticker: string): PredictionResult => {
    const basePrice = Math.random() * 200 + 50; // Random price between $50-$250
    const dates = [];
    const data: StockData[] = [];
    
    // Generate historical data (last 30 days)
    for (let i = 29; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      dates.push(date.toISOString().split('T')[0]);
    }
    
    // Generate future dates (next 10 days)
    for (let i = 1; i <= 10; i++) {
      const date = new Date();
      date.setDate(date.getDate() + i);
      dates.push(date.toISOString().split('T')[0]);
    }
    
    // Generate price data with some volatility
    let currentPrice = basePrice;
    dates.forEach((date, index) => {
      const volatility = (Math.random() - 0.5) * 10; // Random change between -$5 to +$5
      currentPrice = Math.max(currentPrice + volatility, 10); // Ensure price doesn't go below $10
      
      data.push({
        date: date,
        price: Number(currentPrice.toFixed(2)),
        type: index < 30 ? 'historical' : 'predicted'
      });
    });

    const trends = ['bullish', 'bearish', 'neutral'] as const;
    const randomTrend = trends[Math.floor(Math.random() * trends.length)];
    
    return {
      ticker: ticker.toUpperCase(),
      data,
      prediction: {
        trend: randomTrend,
        confidence: Math.random() * 0.4 + 0.6, // 60-100% confidence
        reasoning: `Based on technical analysis and market sentiment, ${ticker.toUpperCase()} shows ${randomTrend} indicators. Key factors include recent trading volume, moving averages, and sector performance trends.`
      }
    };
  };

  const transformApiData = (apiResponse: any): PredictionResult => {
    console.log('Raw API Response:', apiResponse);
    
    const { data } = apiResponse;
    const { historical_prices, predicted_prices, ticker } = data;
    
    // Combine historical and predicted data
    const combinedData: StockData[] = [
      ...historical_prices.map((item: any) => ({
        date: item.date,
        price: Number(item.price),
        type: 'historical' as const
      })),
      ...predicted_prices.map((item: any) => ({
        date: item.date,
        price: Number(item.price),
        type: 'predicted' as const
      }))
    ];

    // Sort by date to ensure proper order
    combinedData.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

    // Calculate trend based on last historical vs first predicted price
    const lastHistorical = historical_prices[historical_prices.length - 1];
    const firstPredicted = predicted_prices[0];
    const priceDiff = firstPredicted.price - lastHistorical.price;
    const percentChange = (priceDiff / lastHistorical.price) * 100;
    
    let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (percentChange > 1) trend = 'bullish';
    else if (percentChange < -1) trend = 'bearish';
    
    // Calculate confidence based on price volatility
    const confidence = Math.min(0.95, Math.max(0.65, 0.8 - Math.abs(percentChange) * 0.01));

    const result: PredictionResult = {
      ticker: ticker,
      data: combinedData,
      prediction: {
        trend,
        confidence,
        reasoning: `Analysis of ${ticker} shows a ${trend} trend with ${percentChange > 0 ? 'an increase' : 'a decrease'} of ${Math.abs(percentChange).toFixed(2)}% expected over the next ${predicted_prices.length} trading days based on ${historical_prices.length} days of historical data.`
      }
    };

    console.log('Transformed Data:', result);
    return result;
  };

  const fetchStockData = async () => {
    if (!ticker.trim()) {
      setError('Please enter a stock ticker');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      // Try to fetch from your API first
      try {
        const response = await fetch('http://localhost:5000/api/receive_data', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ticker: ticker.toUpperCase() }),
        });
        
        if (response.ok) {
          const apiData = await response.json();
          console.log('API Response received:', apiData);
          
          // Transform the API data to match our component structure
          const transformedData = transformApiData(apiData);
          setPrediction(transformedData);
          return;
        }
      } catch (apiError) {
        console.log('API not available, using mock data');
      }
      
      // If API fails, use mock data for demonstration
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate API delay
      const mockData = generateMockData(ticker);
      setPrediction(mockData);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate prediction');
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    await fetchStockData();
  };

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'bullish': return 'text-green-400';
      case 'bearish': return 'text-red-400';
      default: return 'text-yellow-400';
    }
  };

  const getLineColor = (dataPoint: StockData) => {
    return dataPoint.type === 'predicted' ? '#A855F7' : '#8B5CF6';
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
          <p className="text-gray-300">Get AI-powered stock price predictions with interactive charts</p>
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
              onClick={handlePredict}
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
            
            <p className="text-xs text-gray-400 mt-2 text-center">
              Note: If API is unavailable, demo data will be shown
            </p>
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
                    {formatPrice(prediction.data.filter(d => d.type === 'historical').slice(-1)[0]?.price || 0)}
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
                      formatter={(value: number, name: string, props: any) => [
                        formatPrice(value), 
                        props.payload.type === 'predicted' ? 'Predicted Price' : 'Historical Price'
                      ]}
                      labelFormatter={(date: string) => `Date: ${date}`}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      stroke="#8B5CF6"
                      strokeWidth={2}
                      dot={(props: any) => {
                        const { payload } = props;
                        return (
                          <circle
                            cx={props.cx}
                            cy={props.cy}
                            r={3}
                            fill={payload.type === 'predicted' ? '#A855F7' : '#8B5CF6'}
                            strokeWidth={2}
                            stroke={payload.type === 'predicted' ? '#A855F7' : '#8B5CF6'}
                          />
                        );
                      }}
                      activeDot={{ r: 6, fill: '#A855F7' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 flex items-center justify-center gap-8 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-purple-500 rounded"></div>
                  <span className="text-gray-300">Historical Data</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-purple-400 rounded"></div>
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