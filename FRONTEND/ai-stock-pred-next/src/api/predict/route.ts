// src/app/api/predict/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { stockPredictionFlow } from '../../../ai/config';

export async function POST(request: NextRequest) {
  try {
    const { ticker } = await request.json();

    if (!ticker || typeof ticker !== 'string') {
      return NextResponse.json(
        { error: 'Valid ticker symbol is required' },
        { status: 400 }
      );
    }

    // Validate ticker format (basic validation)
    const tickerRegex = /^[A-Z]{1,5}$/;
    if (!tickerRegex.test(ticker)) {
      return NextResponse.json(
        { error: 'Invalid ticker format. Use 1-5 uppercase letters (e.g., AAPL)' },
        { status: 400 }
      );
    }

    // Run the stock prediction flow
    const result = await stockPredictionFlow({ ticker });

    return NextResponse.json(result);
  } catch (error) {
    console.error('Stock prediction error:', error);
    
    return NextResponse.json(
      { 
        error: 'Failed to generate stock prediction',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'Stock Prediction API',
    usage: 'POST /api/predict with { ticker: "AAPL" }'
  });
}