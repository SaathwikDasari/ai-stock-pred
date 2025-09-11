// src/ai/config.ts
import { configureGenkit } from '@genkit-ai/core';
import { googleAI } from '@genkit-ai/googleai';
import { defineFlow, defineSchema } from '@genkit-ai/core';
import { z } from 'zod';

// Configure Genkit
configureGenkit({
  plugins: [
    googleAI({
      apiKey: process.env.GOOGLE_AI_API_KEY,
    }),
  ],
  enableTracingAndMetrics: true,
});

// Schema for stock prediction input
const StockPredictionInput = defineSchema(
  'StockPredictionInput',
  z.object({
    ticker: z.string().describe('Stock ticker symbol (e.g., AAPL, GOOGL)'),
    historicalData: z.array(
      z.object({
        date: z.string(),
        price: z.number(),
      })
    ).optional().describe('Historical stock price data'),
  })
);

// Schema for stock prediction output
const StockPredictionOutput = defineSchema(
  'StockPredictionOutput',
  z.object({
    ticker: z.string(),
    data: z.array(
      z.object({
        date: z.string(),
        price: z.number(),
        type: z.enum(['historical', 'predicted']),
      })
    ),
    prediction: z.object({
      trend: z.enum(['bullish', 'bearish', 'neutral']),
      confidence: z.number().min(0).max(1),
      reasoning: z.string(),
    }),
  })
);

// Generate mock historical data for demonstration
function generateMockHistoricalData(ticker: string) {
  const data = [];
  const basePrice = Math.random() * 200 + 50;
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - 30);

  for (let i = 0; i < 30; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    const price = basePrice + (Math.random() - 0.5) * 20 + Math.sin(i / 5) * 10;
    data.push({
      date: date.toISOString().split('T')[0],
      price: Math.max(10, price),
      type: 'historical' as const
    });
  }
  return data;
}

// Define the stock prediction flow
export const stockPredictionFlow = defineFlow(
  {
    name: 'stockPrediction',
    inputSchema: StockPredictionInput,
    outputSchema: StockPredictionOutput,
  },
  async (input) => {
    const { ticker } = input;
    
    // Generate or use provided historical data
    const historicalData = input.historicalData || generateMockHistoricalData(ticker);
    
    // Prepare the prompt for Gemini
    const prompt = `
      You are a professional stock market analyst with expertise in technical analysis and market trends.
      
      Analyze the following stock: ${ticker}
      Historical price data (last 30 days):
      ${historicalData.map(d => `${d.date}: $${d.price.toFixed(2)}`).join('\n')}
      
      Based on this data, please:
      1. Predict the stock price for the next 14 days
      2. Determine if the overall trend is bullish, bearish, or neutral
      3. Provide a confidence level (0.0 to 1.0)
      4. Give detailed reasoning for your analysis
      
      Please respond in the following JSON format:
      {
        "predictions": [
          {"days_ahead": 1, "predicted_price": 150.25},
          {"days_ahead": 2, "predicted_price": 151.50},
          ... (for 14 days)
        ],
        "trend": "bullish" | "bearish" | "neutral",
        "confidence": 0.75,
        "reasoning": "Detailed analysis explanation..."
      }
    `;

    // Use Gemini AI to generate prediction
    const { generate } = await import('@genkit-ai/ai');
    const { gemini15Pro } = await import('@genkit-ai/googleai');

    const response = await generate({
      model: gemini15Pro,
      prompt: prompt,
      config: {
        temperature: 0.3,
        maxOutputTokens: 2000,
      },
    });

    let aiAnalysis;
    try {
      // Parse the AI response
      const responseText = response.text();
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        aiAnalysis = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error('No valid JSON found in response');
      }
    } catch (error) {
      // Fallback to simulated analysis if parsing fails
      console.warn('Failed to parse AI response, using fallback:', error);
      aiAnalysis = generateFallbackAnalysis(historicalData);
    }

    // Generate predicted data points
    const lastPrice = historicalData[historicalData.length - 1].price;
    const predictedData = aiAnalysis.predictions.map((pred: any, index: number) => {
      const date = new Date();
      date.setDate(date.getDate() + pred.days_ahead);
      return {
        date: date.toISOString().split('T')[0],
        price: pred.predicted_price || (lastPrice * (1 + (Math.random() - 0.5) * 0.1)),
        type: 'predicted' as const
      };
    });

    return {
      ticker,
      data: [...historicalData, ...predictedData],
      prediction: {
        trend: aiAnalysis.trend || 'neutral',
        confidence: aiAnalysis.confidence || 0.5,
        reasoning: aiAnalysis.reasoning || 'Analysis based on historical price patterns and market trends.',
      },
    };
  }
);

// Fallback analysis function
function generateFallbackAnalysis(historicalData: any[]) {
  const prices = historicalData.map(d => d.price);
  const recentPrices = prices.slice(-7); // Last 7 days
  const avgRecent = recentPrices.reduce((a, b) => a + b, 0) / recentPrices.length;
  const avgAll = prices.reduce((a, b) => a + b, 0) / prices.length;
  
  const trend = avgRecent > avgAll * 1.02 ? 'bullish' : 
                avgRecent < avgAll * 0.98 ? 'bearish' : 'neutral';
  
  const predictions = Array.from({ length: 14 }, (_, i) => ({
    days_ahead: i + 1,
    predicted_price: avgRecent * (1 + (Math.random() - 0.5) * 0.1)
  }));

  return {
    predictions,
    trend,
    confidence: 0.6,
    reasoning: `Based on technical analysis, the stock shows a ${trend} trend. Recent 7-day average ($${avgRecent.toFixed(2)}) compared to 30-day average ($${avgAll.toFixed(2)}) indicates ${trend === 'bullish' ? 'upward momentum' : trend === 'bearish' ? 'downward pressure' : 'sideways movement'}.`
  };
}