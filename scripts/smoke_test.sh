#!/bin/bash

set -e

echo "🚀 Starting smoke test..."

# Set test environment
export ENVIRONMENT=${ENVIRONMENT:-test}
export DATABASE_PATH=/tmp/furniture_ai_test.sqlite3
export SERVICE_API_KEY="test-key-for-smoke-testing-only"

# Clean up any previous test database
rm -f $DATABASE_PATH

# Start the server in the background
echo "📋 Starting FastAPI server..."
python -m uvicorn furniture_ai.api:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to be ready (max 30 seconds)..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ Server is ready!"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "❌ Server failed to start within 30 seconds"
    kill $SERVER_PID || true
    exit 1
  fi
  sleep 1
done

# Test /health endpoint
echo ""
echo "🔍 Testing /health endpoint..."
HEALTH_RESPONSE=$(curl -s http://127.0.0.1:8000/health)
echo "Response: $HEALTH_RESPONSE"

if echo "$HEALTH_RESPONSE" | grep -q '"status"'; then
  echo "✅ /health endpoint OK"
else
  echo "❌ /health endpoint failed"
  kill $SERVER_PID
  exit 1
fi

# Test /ready endpoint
echo ""
echo "🔍 Testing /ready endpoint..."
READY_RESPONSE=$(curl -s http://127.0.0.1:8000/ready)
echo "Response: $READY_RESPONSE"

if echo "$READY_RESPONSE" | grep -q '"status"'; then
  echo "✅ /ready endpoint OK"
else
  echo "❌ /ready endpoint failed"
  kill $SERVER_PID
  exit 1
fi

# Clean up
echo ""
echo "🧹 Shutting down server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "✅ All smoke tests passed!"
exit 0
