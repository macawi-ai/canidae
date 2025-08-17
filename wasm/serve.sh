#!/bin/bash

echo "🐺 Starting CANIDAE WASM Demo Server"
echo "================================================"
echo ""
echo "1. Starting WebSocket test server on ws://localhost:8080/pack"
python3 test-server.py &
WS_PID=$!
echo "   WebSocket server PID: $WS_PID"
echo ""

echo "2. Starting HTTP server on http://localhost:8000"
echo "   Open http://localhost:8000 in your browser"
echo ""
python3 -m http.server 8000

# Clean up WebSocket server when HTTP server stops
kill $WS_PID 2>/dev/null
echo "Servers stopped"