#!/bin/bash
# Launch Streamlit Dashboard

cd /home/yeblad/Desktop/Geospatial_Crime_Analysis

echo "=================================================="
echo "  FBI Crime Pattern Analysis Dashboard"
echo "=================================================="
echo ""
echo "Starting Streamlit server..."
echo "Dashboard will open in your browser automatically"
echo ""
echo "To stop: Press Ctrl+C"
echo "=================================================="
echo ""

streamlit run streamlit_app.py --server.port 8501 --server.address localhost
