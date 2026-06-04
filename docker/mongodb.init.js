// Grey MongoDB Initialization Script
// This script runs when the container first starts

// Switch to admin database
db = db.getSiblingDB('admin');

// Create application database
db = db.getSiblingDB('grey_backtest');

// Create collections with indexes
db.createCollection('strategies');
db.createCollection('optimize_history');
db.createCollection('backtest_results');

// Create indexes for performance
db.strategies.createIndex({ batch_id: 1, status: 1 });
db.strategies.createIndex({ 'params.length_ema': 1, 'params.length_atr': 1 });
db.strategies.createIndex({ 'metrics.roi': -1 });
db.strategies.createIndex({ 'metrics.sharpe': -1 });
db.strategies.createIndex({ created_at: -1 });

db.optimize_history.createIndex({ batch_id: 1 });
db.optimize_history.createIndex({ status: 1 });
db.optimize_history.createIndex({ created_at: -1 });

db.backtest_results.createIndex({ asset: 1, timeframe: 1 });
db.backtest_results.createIndex({ created_at: -1 });

print('✅ Grey MongoDB initialized successfully');
print('📊 Collections created: strategies, optimize_history, backtest_results');
print('🔍 Indexes created for optimal query performance');
