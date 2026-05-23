# ATLAS API Test Guide

## Health
curl -s http://localhost:8080/health | jq .

## Strategies
curl -s http://localhost:8080/api/strategies | jq 'length'

## Live Market Data
curl -s "http://localhost:8080/api/market-data?symbol=BTC%2FUSDT&limit=5" | jq .

## Pattern Detection
curl -s -X POST http://localhost:8080/api/patterns/detect \
-H "Content-Type: application/json" \
-d '{"symbol":"AAPL","lookback_bars":120}' | jq .

## Self Improvement
curl -s -X POST http://localhost:8080/api/self-improvement/run-cycle \
-H "Content-Type: application/json" \
-d '{"limit":10}' | jq .

## Copy Trading
curl -s -X POST http://localhost:8080/api/copy-trading/mirror-test \
-H "Content-Type: application/json" \
-d '{"leader_account":"leader_latency_test","leader_order_id":"copy-test","symbol":"AAPL","side":"buy","leader_qty":10,"leader_equity":100000,"follower_equity":25000,"price":100,"fill_ratio":0.5}' | jq .

## Dashboard
curl -I http://localhost:8080/dashboard/
